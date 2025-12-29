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
from torch.nn.parallel import DistributedDataParallel as DDP
# --- Fairscale specific imports ---
try:
    from fairscale.nn.model_parallel.initialize import (
        initialize_model_parallel,
        get_model_parallel_rank,
        get_model_parallel_world_size,
        get_model_parallel_group,
        model_parallel_is_initialized,
        get_data_parallel_world_size,
        get_data_parallel_rank
    )
    from fairscale.nn.model_parallel.cross_entropy import vocab_parallel_cross_entropy # Added this line
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
    params_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 1,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    log_interval: int = 10,
    model_parallel_size: int = 2,
):
    start_time = time.time()

    """
    Model Parallel (Tensor Parallel) training script using Fairscale.
    """
    setup(model_parallel_size)

    local_rank = int(os.environ["LOCAL_RANK"])
    mp_rank = get_model_parallel_rank()
    world_size = dist.get_world_size()
    
    device = f"cuda:{local_rank}"
    
    if is_main_process():
        print(f"---- Initializing Model with Model Parallel Size: {model_parallel_size} ----")

    with open(params_path, "r") as f:
        params = json.load(f)
    
    model_args = ModelArgs(**params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokenizer.pad_id = tokenizer.eos_id
    model_args.vocab_size = tokenizer.n_words
    
    model = Transformer(model_args)

    shard_path = os.path.join(ckpt_dir, f"mp_rank_{mp_rank:02d}.pth")
    if os.path.exists(shard_path):
        print(f"Info (Rank {dist.get_rank()}): Loading model shard from {shard_path}")
        checkpoint = torch.load(shard_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(f"Warning (Rank {dist.get_rank()}): No model shard found at {shard_path}. Training from scratch.")
    
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataset = PubMedQADataset(data_path, tokenizer, model_args.max_seq_len)
    # sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=dist.get_rank(), drop_last=True)
    sampler = DistributedSampler(
        dataset, 
        shuffle=True, 
        num_replicas=get_data_parallel_world_size(), # TP 2개로 모델 1개만 만들었다면 이 값은 1이 됩니다.
        rank=get_data_parallel_rank(),               # TP 그룹 내의 모든 rank는 여기서 동일한 0을 가집니다.
        drop_last=True
    )
    print("get_data_parallel_world_size ",get_data_parallel_world_size())
    print("get_data_parallel_rank ", get_data_parallel_rank())
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id)
    )
    print(f"Model and tokenizer loaded in {time.time() - start_time:.2f}s")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        if is_main_process():
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs, start_pos=0)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_id)
            # loss = vocab_parallel_cross_entropy(logits, targets, ignore_index=tokenizer.pad_id)                                             

            losses = vocab_parallel_cross_entropy(logits, targets)

            # 2. 정답(targets)에서 패딩이 아닌 부분만 1, 패딩인 부분은 0인 마스크 생성
            mask = (targets != tokenizer.pad_id)
            local_valid_tokens = mask.sum()

            # 🔥 batch 단위 rank 동기화 (모든 rank가 동일한 판단을 하도록)
            valid_tokens = torch.tensor(
                local_valid_tokens.item(),
                device=device,
                dtype=torch.int64
            )
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.MIN)

            # 모든 rank에서 동일하게 skip
            if valid_tokens.item() == 0:
                optimizer.zero_grad(set_to_none=True)
                if is_main_process():
                    print(f"[Epoch {epoch+1}] Skipping batch {i} (all pad)")
                continue

            # 3. 마스크를 곱해서 패딩 위치의 Loss를 0으로 만듦
            losses = losses * mask

            # 4. 전체 평균 계산 (실제 유효한 토큰 개수로 나누기)
            loss = losses.sum() / mask.sum()                                                             
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()

            if i % log_interval == 0 and is_main_process():
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss_tensor = torch.tensor(total_loss / len(dataloader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

        if is_main_process():
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # if mp_rank == 0:
    print(f"---- Saving Model Shard on Global Rank {dist.get_rank()} ----")
    os.makedirs(output_dir, exist_ok=True)
    shard_name = f"llama3_pubmedqa_mp_shard_rank_{mp_rank:02d}.pth"
    model_save_path = os.path.join(output_dir, shard_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model shard saved to {model_save_path}")
    print(f"Train Exit in {time.time() - start_time:.2f}s")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Parallel Llama3 Training Script using Fairscale")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the sharded model checkpoints.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.model file.")
    parser.add_argument("--params_path", type=str, required=True, help="Path to the params.json file from the original model directory.")
    parser.add_argument("--data_path", type=str, default="pubmeqa_preprocessed.txt", help="Path to the data file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llama3-8b-pubmedqa-mp", help="Directory to save fine-tuned model shards.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size PER DATA PARALLEL REPLICA.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging.")
    parser.add_argument("--model_parallel_size", type=int, default=2, help="Number of GPUs to use for model parallelism.")
    
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        params_path=args.params_path,
        output_dir=args.output_dir,
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        model_parallel_size=args.model_parallel_size,
    )
