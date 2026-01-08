import os
import random
import zarr
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist

class InfiniteZarrDataset(IterableDataset):
    def __init__(self, data_root, crop_size=(64, 64, 64), resolution_key='s0'):
        """
        Args:
            data_root (str): Path to root directory containing volume folders.
            crop_size (tuple): (D, H, W) size of the crop.
            resolution_key (str): The specific resolution array to target (e.g., 's0', 's1').
        """
        super().__init__()
        self.data_root = data_root
        self.crop_size = crop_size
        self.resolution_key = resolution_key
        
        # Pre-scan for valid volumes to avoid hitting disk repeatedly
        self.volume_paths = self._find_all_volumes()
        if not self.volume_paths:
            raise FileNotFoundError(f"No .zarr volumes found in {data_root}")

    def _find_all_volumes(self):
        """Finds all .zarr directories following the data/name/name.zarr pattern."""
        volumes = []
        if not os.path.exists(self.data_root):
            return []
            
        for vol_name in os.listdir(self.data_root):
            # Structure: data/vol_name/vol_name.zarr
            vol_dir = os.path.join(self.data_root, vol_name)
            if os.path.isdir(vol_dir):
                potential_zarr = os.path.join(vol_dir, f"{vol_name}.zarr")
                if os.path.exists(potential_zarr):
                    volumes.append(potential_zarr)

        return volumes

    def _get_array_from_volume(self, vol_path):
        """
        Dynamically traverses the variable directory structure:
        root -> recon-* -> em -> fibsem-* -> resolution_key
        """
        try:
            # Open in read mode
            store = zarr.DirectoryStore(vol_path)
            root = zarr.group(store=store, mode='r')

            # 1. Find recon folder (handle variable names like recon-1, recon-2)
            recon_key = next((k for k in root.keys() if k.startswith('recon-')), None)
            if recon_key is None: return None
            
            # 2. Find em folder
            if 'em' not in root[recon_key]: return None
            em_group = root[recon_key]['em']

            # 3. Find fibsem folder (handle variable names like fibsem-uint8, fibsem-int16)
            fibsem_key = next((k for k in em_group.keys() if k.startswith('fibsem-')), None)
            if fibsem_key is None: return None
            
            # 4. Return the specific resolution array
            fibsem_group = em_group[fibsem_key]
            if self.resolution_key in fibsem_group:
                return fibsem_group[self.resolution_key]
                
        except Exception as e:
            # In production, use logging.warning here
            pass

        return None

    def __iter__(self):
        """
        The core infinite loop. We initialize RNG here to ensure DDP workers are unique.
        """
        # --- DDP & Worker Seeding ---
        worker_info = torch.utils.data.get_worker_info()
        
        # Calculate a unique seed based on rank and worker_id
        # If not using DDP, rank is 0.
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

        worker_id = worker_info.id if worker_info is not None else 0
        
        # Create a unique seed: (global_rank * 1000) + worker_id
        # This ensures GPU 0 Worker 0 is different from GPU 1 Worker 0
        seed = (rank * 1000) + worker_id 
        
        # Seed the RNGs
        random.seed(seed)
        np.random.seed(seed)
        
        # --- Infinite Loop ---
        while True:
            # 1. Pick a random volume
            vol_path = random.choice(self.volume_paths)
            
            # 2. Open the array (handle structure)
            data_array = self._get_array_from_volume(vol_path)
            
            # If volume structure was invalid or broken, try again immediately
            if data_array is None:
                continue

            # 3. Determine Shapes
            # Zarr shape is typically (z, y, x)
            z_shape, y_shape, x_shape = data_array.shape
            c_z, c_y, c_x = self.crop_size

            # Skip if volume is smaller than crop
            if z_shape < c_z or y_shape < c_y or x_shape < c_x:
                continue

            # 4. Select Random Coordinates
            z_start = random.randint(0, z_shape - c_z)
            y_start = random.randint(0, y_shape - c_y)
            x_start = random.randint(0, x_shape - c_x)

            # 5. Load Data (Slicing Zarr reads only the specific chunk)
            try:
                crop = data_array[
                    z_start : z_start + c_z,
                    y_start : y_start + c_y,
                    x_start : x_start + c_x
                ]
                
                # Preprocessing (Normalization, etc. can go here)
                # Ensure float32 for training
                tensor = torch.from_numpy(crop).float()
                
                # Add channel dimension: (D, H, W) -> (C, D, H, W)
                tensor = tensor.unsqueeze(0) 

                yield tensor

            except Exception:
                # Handle read errors (e.g. network glitch on mounted drive)
                continue

def get_infinite_dataloader(
    data_root, 
    batch_size, 
    crop_size=(8, 8, 8),
    resolution_key='s0',
    num_workers=4
):
    """
    Factory function to create the loader with configurable crop size.
    """
    dataset = InfiniteZarrDataset(
        data_root=data_root,
        crop_size=crop_size,       # Pass the argument here
        resolution_key=resolution_key
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )