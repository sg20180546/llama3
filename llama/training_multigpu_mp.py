# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
#
# This script is modified to support Tensor Parallelism using Fairscale
# for model parallel training.

import json
import os
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# --- Fairscale specific imports ---
try:
    from fairscale.nn.model_parallel.initialize import (
        initialize_model_parallel,
        get_model_parallel_rank,
        get_model_parallel_world_size,
        get_model_parallel_group,
        model_parallel_is_initialized,
    )
    FAIRSCALE_AVAILABLE = True
except ImportError:
    FAIRSCALE_AVAILABLE = False
# --- End Fairscale imports ---

from model import ModelArgs, Transformer
from tokenizer import Tokenizer

def setup(model_parallel_size: int):
    """Sets up the distributed environment for model parallelism."""
    if not FAIRSCALE_AVAILABLE:
        raise RuntimeError("Fairscale is not available. Please install it to use model parallelism.")

    dist.init_process_group("nccl")
    # Initialize model parallel group
    initialize_model_parallel(model_parallel_size)
    
    # Set the device for the current process
    torch.cuda.set_device(get_model_parallel_rank())

def cleanup():
    """Cleans up the distributed environment."""
    if model_parallel_is_initialized():
        dist.destroy_process_group(get_model_parallel_group())
    dist.destroy_process_group()

def is_main_process():
    # In model parallelism, rank 0 of the model parallel group is often considered the main process
    # for logging and saving. We also check the global rank.
    return dist.get_rank() == 0

class PubMedQADataset(Dataset):
    """PyTorch Dataset for PubMedQA."""
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if is_main_process():
            print("Loading training data...")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.read().strip().split('\n\n')
        if is_main_process():
            print(f"Loaded {len(self.data)} training entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text, bos=True, eos=True)
        
        # Truncate if longer than max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch: List[torch.Tensor], pad_id: int):
    """Pad sequences to the max length in a batch."""
    max_len = max(len(t) for t in batch)
    padded_tokens = [F.pad(t, (0, max_len - len(t)), value=pad_id) for t in batch]
    tokens = torch.stack(padded_tokens)
    
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    return inputs, targets

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 1,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    log_interval: int = 10,
    model_parallel_size: int = 2,
):
    """
    Model Parallel (Tensor Parallel) training script using Fairscale.
    """
    setup(model_parallel_size)

    # Each process has a local rank within its node, and a global rank.
    # The model parallel rank determines which part of the model it holds.
    local_rank = int(os.environ["LOCAL_RANK"])
    mp_rank = get_model_parallel_rank()
    world_size = dist.get_world_size() # This is the global world size
    
    device = f"cuda:{local_rank}"
    
    # ---- 1. Load Model and Tokenizer ----
    if is_main_process():
        print(f"---- Initializing Model and Tokenizer with Model Parallel Size: {model_parallel_size} ----")

    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    model_args = ModelArgs(**params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokenizer.pad_id = tokenizer.eos_id
    model_args.vocab_size = tokenizer.n_words
    
    # ---- 2. Prepare Model for Tensor Parallelism ----
    # Fairscale's initialize_model_parallel() enables the logic within the model
    # to use parallel layers. We just need to instantiate it.
    model = Transformer(model_args)

    # Load checkpoint shards if available.
    # Note: For tensor parallel models, checkpoints are sharded. This is a simplified
    # loading logic that assumes a non-sharded checkpoint to start from.
    checkpoint_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    if checkpoint_paths:
        # For simplicity, we load the full checkpoint on all processes, and Fairscale's
        # parallel layers will only take their slice of the weights.
        if is_main_process():
            print(f"Info: Loading checkpoint from {checkpoint_paths[0]} for fine-tuning.")
        checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        torch.cuda.empty_cache()
    else:
        if is_main_process():
            print("Info: No checkpoint found. Training from scratch.")
    
    # Move model to the correct device. With Fairscale MP, different layers
    # are on different devices already. We move the non-parallel parts.
    model.to(device)

    if is_main_process():
        # A simple way to show memory distribution
        print("--- Model Memory Allocation (GPU 0) ---")
        os.system(f"nvidia-smi --query-gpu=memory.used --format=csv -i 0")


    # ---- 3. Prepare Optimizer and Data ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Data is distributed across all processes (data parallel dimension, which is 1 here).
    # Each process gets a slice of the data.
    dataset = PubMedQADataset(data_path, tokenizer, model_args.max_seq_len)
    sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=dist.get_rank())
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id)
    )
         
    # ---- 4. Training Loop ----
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        if is_main_process():
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs, start_pos=0)
            
            # Loss calculation is the same, but it's computed on sharded outputs.
            # The parallel cross-entropy loss would be more efficient, but this works for a start.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()

            if i % log_interval == 0 and is_main_process():
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Average loss across all processes
        avg_loss_tensor = torch.tensor(total_loss / len(dataloader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

        if is_main_process():
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # ---- 5. Save Model ----
    # For model parallelism, we save a sharded checkpoint.
    # Only the main process of the model parallel group saves its shard.
    if get_model_parallel_rank() == 0:
        print(f"---- Saving Model Shard on Rank {dist.get_rank()} ----")
        os.makedirs(output_dir, exist_ok=True)
        # Name the shard based on the model parallel rank
        shard_name = f"llama3_pubmedqa_mp_shard_rank_{get_model_parallel_rank()}.pth"
        model_save_path = os.path.join(output_dir, shard_name)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model shard saved to {model_save_path}")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Parallel Llama3 Training Script using Fairscale")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.model file")
    parser.add_argument("--data_path", type=str, default="pubmeqa_preprocessed.txt", help="Path to the data file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llama3-8b-pubmedqa-mp", help="Directory to save model shards")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size PER DATA PARALLEL REPLICA")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--model_parallel_size", type=int, default=2, help="Number of GPUs to use for model parallelism")
    
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        model_parallel_size=args.model_parallel_size,
    )
