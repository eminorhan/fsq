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


def get_lr_lambda(current_step, warmup_steps, train_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)))


if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()

    CHECKPOINT_DIR = 'ckpts'
    CHECKPOINT_INTERVAL = 50_000
    LOG_INTERVAL = 1_000
    LOAD_CHECKPOINT_PATH = None 

    # Data Params
    BATCH_SIZE = 64
    INPUT_DIM = np.prod(PATCH_SIZE)

    # FSQ Levels
    levels = [8, 8, 7, 7, 6, 6] 

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

    # Model initialization
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
                }, save_path)
                print(f"Checkpoint saved: {save_path}")

    cleanup_distributed()