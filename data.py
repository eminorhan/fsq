import torch
from torch.utils.data import IterableDataset
import numpy as np
import zarr
from pathlib import Path


def get_rank_and_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def discover_volumes(data_root: Path):
    return sorted(p for p in data_root.rglob("*.zarr") if p.is_dir())


def find_resolution_array(zarr_root: Path, resolution: str):
    for path in zarr_root.rglob(resolution):
        try:
            arr = zarr.open(path, mode="r")
            if isinstance(arr, zarr.core.Array) and arr.ndim == 3:
                return arr
        except Exception:
            continue
    raise FileNotFoundError(f"Resolution {resolution} not found in {zarr_root}")


class ZarrRandomSubvolumeDataset(IterableDataset):
    """
    Infinite iterable dataset that randomly samples 3D subvolumes from a collection of Zarr EM volumes.
    """
    def __init__(
        self,
        data_root,
        patch_size,            # (pz, py, px)
        resolution="s0",
        seed=0,
        return_metadata=False,
        dtype=torch.float32,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.patch_size = tuple(patch_size)
        self.resolution = resolution
        self.seed = seed
        self.return_metadata = return_metadata
        self.dtype = dtype

        self.volumes = discover_volumes(self.data_root)
        if len(self.volumes) == 0:
            raise RuntimeError(f"No .zarr volumes found under {data_root}")

    def _make_rng(self):
        rank, _ = get_rank_and_world_size()
        return np.random.default_rng(self.seed + rank)

    def _sample_patch(self, rng):
        # 1. Random volume
        vol_path = self.volumes[rng.integers(len(self.volumes))]
        vol_name = vol_path.stem.replace(".zarr", "")

        # # print (for debugging purposes)
        # print(f"[Rank: {torch.distributed.get_rank()}] Sampling volume: {vol_name}")
        
        # 2. Resolution array
        arr = find_resolution_array(vol_path, self.resolution)

        z, y, x = arr.shape
        pz, py, px = self.patch_size

        if pz > z or py > y or px > x:
            raise ValueError(f"Patch {self.patch_size} larger than volume {arr.shape}")

        # 3. Random crop
        z0 = rng.integers(0, z - pz + 1)
        y0 = rng.integers(0, y - py + 1)
        x0 = rng.integers(0, x - px + 1)

        patch = arr[z0:z0+pz, y0:y0+py, x0:x0+px]

        meta = {
            "volume": vol_name,
            "zarr_path": str(vol_path),
            "resolution": self.resolution,
            "offset": (int(z0), int(y0), int(x0)),
        }
        return patch, meta

    def __iter__(self):
        rng = self._make_rng()

        while True:
            patch, meta = self._sample_patch(rng)

            patch = torch.from_numpy(np.asarray(patch) / 12000.0).to(self.dtype) 

            patch = patch.flatten()
            # print(f"patch shape: {patch.shape}")

            if self.return_metadata:
                yield patch, meta
            else:
                yield patch
