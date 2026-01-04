import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from torch.utils.data import IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt

from model import TransformerFSQ

def setup_distributed():
    dist.init_process_group(backend="nccl")
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"Distributed setup: Rank {rank}/{world_size} on device {local_rank}")
    return rank, world_size, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

# --- Data Utils ---
def get_patches_column_major(data_array: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """Extracts patches in column-major order."""
    if data_array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional, but got {data_array.ndim}")
    p0, p1 = patch_size
    n, t = data_array.shape
    pad_n = (p0 - (n % p0)) % p0
    pad_t = (p1 - (t % p1)) % p1

    if pad_n > 0 or pad_t > 0:
        data_array = np.pad(data_array, ((0, pad_n), (0, pad_t)), mode='constant', constant_values=0)
    
    N, T = data_array.shape
    num_patches_n = N // p0
    num_patches_t = T // p1

    reshaped = data_array.reshape(num_patches_n, p0, num_patches_t, p1)
    transposed = reshaped.transpose(2, 0, 1, 3)
    patches = transposed.reshape(-1, p0*p1)
    
    patches = np.unique(patches, axis=0)
    np.random.shuffle(patches)
    return patches

class IterablePatchDataset(IterableDataset, Stateful):
    def __init__(self, dataset_name: str, patch_size: Tuple[int, int], world_size: int = 1, rank: int = 0) -> None:
        ds = load_dataset(dataset_name, split="train")
        self._data = split_dataset_by_node(ds, rank, world_size)
        self.dataset_name = dataset_name
        self.rank = rank
        self.patch_size = patch_size
        self._sample_idx = 0

    def __iter__(self):
        while True:
            for samples in self._get_data_iter():
                samples = np.array(samples['spike_counts'])
                # Get patches (flat vectors)
                patches = get_patches_column_major(samples, self.patch_size)
                self._sample_idx += 1
                
                # Yields (Input_Dim)
                for sample in patches:
                    yield torch.logit(torch.from_numpy(sample) / 255.0, eps=1e-6)
            print(f"Dataset {self.dataset_name} is being re-looped on rank {self.rank}")

    def _get_data_iter(self):
        if self._sample_idx == len(self._data):
            return iter([])
        else:
            return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}

class DPAwareDataLoader(StatefulDataLoader, Stateful):
    def __init__(self, dp_rank: int, dataset: IterableDataset, batch_size: int):
        super().__init__(dataset, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict or self._rank_id not in state_dict: return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))

def get_lr_lambda(current_step, warmup_steps, train_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)))


if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()

    CHECKPOINT_DIR = 'outputs_transformer_fsq'
    CHECKPOINT_INTERVAL = 50_000
    LOG_INTERVAL = 1_000
    LOAD_CHECKPOINT_PATH = None 

    # Data Params
    BATCH_SIZE = 64
    PATCH_SIZE = (1, 10)
    INPUT_DIM = np.prod(PATCH_SIZE)

    # FSQ Levels
    levels = [8, 8, 8, 5, 5] 

    # Transformer Hyperparams
    D_MODEL = 256
    NUM_HEADS = 4
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    
    # Training Params
    TRAIN_STEPS = 500_000
    WARMUP_STEPS = 5_000
    LEARNING_RATE = 3e-4

    train_step = 0
    optimizer_state_to_load = None
    scheduler_state_to_load = None

    # Model Initialization (Dropout removed)
    model = TransformerFSQ(
        levels=levels,
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS
    )

    model.to(local_rank)
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Dataset
    ds = IterablePatchDataset("eminorhan/neural-pile-rodent", PATCH_SIZE, world_size, rank)
    dl = DPAwareDataLoader(rank, ds, batch_size=BATCH_SIZE)
    dl_iter = iter(dl)

    if LOAD_CHECKPOINT_PATH and rank == 0 and os.path.exists(LOAD_CHECKPOINT_PATH):
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        ds.load_state_dict(checkpoint['dataset_state_dict'])
        dl.load_state_dict(checkpoint['dataloader_state_dict'])
        train_step = checkpoint['train_step']
        optimizer_state_to_load = checkpoint['optimizer_state_dict']
        scheduler_state_to_load = checkpoint['scheduler_state_dict']
        print(f"Rank 0: Loaded checkpoint from {LOAD_CHECKPOINT_PATH}")

    dist.barrier(device_ids=[local_rank])
    train_step_tensor = torch.tensor([train_step], dtype=torch.long, device=local_rank)
    dist.broadcast(train_step_tensor, src=0)
    train_step = train_step_tensor.item()

    # DDP Wrap
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: get_lr_lambda(s, WARMUP_STEPS, TRAIN_STEPS))
    
    if optimizer_state_to_load: optimizer.load_state_dict(optimizer_state_to_load)
    if scheduler_state_to_load: scheduler.load_state_dict(scheduler_state_to_load)

    if rank == 0: os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    loss_fn = nn.MSELoss()
    model.train()
    running_loss = 0.0

    while train_step < TRAIN_STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        optimizer.zero_grad()
        
        # Batch: (Batch_Size, Input_Dim)
        data = batch.to(local_rank, non_blocking=True)
        data_seq = data.unsqueeze(1) 

        # Forward
        x_hat, _, _ = model(data_seq)
        
        # Loss
        loss = loss_fn(x_hat, data_seq)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        train_step += 1
        
        if rank == 0:
            running_loss += loss.item()
            
            if train_step % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                lr = scheduler.get_last_lr()[0]
                print(f"Step {train_step} | Loss: {avg_loss:.6f} | LR: {lr:.6f}")
                running_loss = 0.0
                
                # Simple visualization
                x_vis = x_hat.detach().cpu().squeeze(1) 
                data_vis = data.detach().cpu()
                
                plt.figure(figsize=(8, 4))
                for i in range(min(4, BATCH_SIZE)):
                    plt.subplot(2, 4, i+1)
                    plt.imshow(data_vis[i].view(PATCH_SIZE[0], PATCH_SIZE[1]), cmap='jet')
                    plt.axis('off')
                    plt.title("Orig")
                    
                    plt.subplot(2, 4, i+5)
                    plt.imshow(x_vis[i].view(PATCH_SIZE[0], PATCH_SIZE[1]), cmap='jet')
                    plt.axis('off')
                    plt.title("Rec")
                plt.tight_layout()
                plt.savefig(f"{CHECKPOINT_DIR}/vis_{train_step}.png")
                plt.close()

            if train_step % CHECKPOINT_INTERVAL == 0:
                save_path = f"{CHECKPOINT_DIR}/ckpt_{train_step}.pth"
                torch.save({
                    'train_step': train_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'dataset_state_dict': ds.state_dict(),
                    'dataloader_state_dict': dl.state_dict(),
                }, save_path)
                print(f"Checkpoint saved: {save_path}")

    cleanup_distributed()