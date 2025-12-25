# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, StateDictType

from model import ModelArgs, Transformer
from tokenizer import Tokenizer

def setup():
    """Sets up the distributed environment."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

class PubMedQADataset(Dataset):
    """PyTorch Dataset for PubMedQA."""
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        print("Loading training data...")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.read().strip().split('\n\n')
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
    
    # Input is all but the last token, target is all but the first
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
):
    """
    Multi-GPU training script for the Llama3 model using DDP.
    """
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    
    # ---- 1. Load Model and Tokenizer (on main process) ----
    if is_main_process():
        print("---- Initializing Model and Tokenizer ----")
        start_time = time.time()

    # Ensure the checkpoint directory exists
    if not os.path.isdir(ckpt_dir):
        if is_main_process():
            print(f"Checkpoint directory {ckpt_dir} not found.")
        return

    # Load model parameters
    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    model_args = ModelArgs(**params)

    # Load tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokenizer.pad_id = tokenizer.eos_id  # Use EOS token for padding
    model_args.vocab_size = tokenizer.n_words
    
    # ---- 2. Prepare Model for DDP ----
    # All processes initialize the model
    model = Transformer(model_args)
    
    # Load checkpoint if available
    checkpoint_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    if checkpoint_paths:
        checkpoint_path = checkpoint_paths[0]
        if is_main_process():
            print(f"Info: Loading checkpoint from {checkpoint_path} for fine-tuning.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        torch.cuda.empty_cache()
    else:
        if is_main_process():
            print("Info: No checkpoint found. Training from scratch.")

    # FSDP handles placing parameters on devices itself.
    # We can provide a param_init_fn to initialize parameters on the correct device.
    # Wrap the model with FSDP
    model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD, param_init_fn=lambda module: module.to(device))

    if is_main_process():
        print(f"Model and tokenizer loaded in {time.time() - start_time:.2f}s")

    # ---- 3. Prepare Optimizer and Data ----
    # Use a memory-efficient optimizer like AdamW8bit if available, otherwise standard AdamW
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
        if is_main_process():
            print("Using 8-bit AdamW optimizer.")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if is_main_process():
            print("bitsandbytes not found. Using standard AdamW optimizer.")


    # Create dataset and distributed sampler
    dataset = PubMedQADataset(data_path, tokenizer, model_args.max_seq_len)
    sampler = DistributedSampler(dataset, shuffle=True)
    
    # Create dataloader
    # The collate_fn needs the pad_id, so we wrap it in a lambda
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id)
    )
         
    # ---- 4. Training Loop ----
    for epoch in range(epochs):
        sampler.set_epoch(epoch) # Necessary for shuffling to work correctly
        model.train()
        total_loss = 0
        
        if is_main_process():
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            logits = model(inputs, start_pos=0)
            
            # Calculate loss
            # The shape of logits is (batch_size, seq_len, vocab_size)
            # The shape of targets is (batch_size, seq_len)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_id)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()

            # Log progress on the main process
            if i % log_interval == 0 and is_main_process():
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Synchronize and calculate average loss across all processes
        dist.all_reduce(torch.tensor(total_loss, device=device), op=dist.ReduceOp.SUM)
        avg_loss = total_loss / len(dataloader)

        if is_main_process():
            print(f"Epoch {epoch+1} Average Loss: {avg_loss / dist.get_world_size():.4f}")


    # ---- 5. Save Model ----
    if is_main_process():
        print("---- Saving Final Model ----")
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, "llama3_pubmedqa_finetuned.pth")
        
        # FSDP uses its own state_dict logic to gather full model parameters on rank 0.
        # We need to set the state_dict_type context for this.
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_model_state = model.state_dict()
            torch.save(full_model_state, model_save_path)
        print(f"Model saved to {model_save_path}")

    cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-GPU Llama3 Training Script for PubMedQA")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the model checkpoint (e.g., 'Meta-Llama-3-8B-Instruct')")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.model file")
    parser.add_argument("--data_path", type=str, default="pubmeqa_preprocessed.txt", help="Path to the preprocessed PubMedQA data file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llama3-8b-pubmedqa", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size PER GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging training loss")
    
    # Note: local_rank is handled by the launch utility (torchrun) 
    
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
    )