# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import time
import torch
from pathlib import Path
from typing import Optional

import torch.nn.functional as F
import torch.nn as nn
from bitsandbytes.optim import AdamW8bit

from accelerate import Accelerator
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 1,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    log_interval: int = 1,
    seed: int = 1,
):
    """
    A simple training script for the Llama3 model, adapted for use with Accelerate.
    """
    # ---- 1. Initialize Accelerator and Basic Setup ----
    start_time = time.time()
    accelerator = Accelerator(mixed_precision='bf16')
    
    # seed must be the same in all processes
    torch.manual_seed(seed)

    # ---- 2. Load Model and Tokenizer ----
    # Ensure the checkpoint directory exists
    if not os.path.isdir(ckpt_dir):
        accelerator.print(f"Checkpoint directory {ckpt_dir} not found.")
        return

    # It's recommended to load model parameters on the main process to avoid OOM.
    # However, for model instantiation, some parameters like vocab_size are needed on all processes.
    
    with open(os.path.join(ckpt_dir, "params.json"), "r") as f:
        params = json.load(f)
    model_args = ModelArgs(**params)

    # Load tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokenizer.pad_id = tokenizer.eos_id
    model_args.vocab_size = tokenizer.n_words
    
    # Instantiate the model
    # All processes need to instantiate the model so that accelerate can handle it
    with accelerator.main_process_first():
        model = Transformer(model_args)

        # Load the checkpoint if it exists
        checkpoint_paths = sorted(Path(ckpt_dir).glob("*.pth"))
        if checkpoint_paths:
            # In a single-GPU or non-sharded setup, we load the first checkpoint.
            # DeepSpeed with ZeRO-3 would handle sharded checkpoint loading automatically.
            checkpoint_path = checkpoint_paths[0]
            accelerator.print(f"Info: Loading checkpoint from {checkpoint_path} for fine-tuning.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        else:
            accelerator.print("Info: No checkpoint file (.pth) found. Initializing model with random weights.")

    accelerator.print(f"Model and tokenizer loaded in {time.time() - start_time:.2f}s")

    # ---- 3. Prepare Optimizer and Data ----
    optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    accelerator.print("Loading training data...")
    with open(data_path, "r", encoding="utf-8") as f:
        train_data = f.read().strip().split('\n\n')
    accelerator.print(f"Loaded {len(train_data)} training entries.")
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Prepare everything with Accelerate
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
         
    # ---- 4. Training Loop ----
    model.train()

    for epoch in range(epochs):
        accelerator.print(f"--- Epoch {epoch+1}/{epochs} ---")
        total_loss = 0
        
        for i, batch_texts in enumerate(train_dataloader):
            # Tokenize and prepare batch on the fly
            batch_tokens = [tokenizer.encode(text, bos=True, eos=True) for text in batch_texts]
            
            for j in range(len(batch_tokens)):
                if len(batch_tokens[j]) > model_args.max_seq_len:
                    batch_tokens[j] = batch_tokens[j][:model_args.max_seq_len]

            max_len = max(len(t) for t in batch_tokens)
            
            padded_tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in batch_tokens]
            tokens = torch.tensor(padded_tokens, dtype=torch.long).to(accelerator.device)
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            # Forward pass
            logits = model(inputs, start_pos=0)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()

            if i % log_interval == 0:
                accelerator.print(f"Batch {i + 1}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # ---- 5. Save Model ----
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, "llama3_finetuned_model.pth")
        
        # Unwrap the model before saving
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), model_save_path)
        accelerator.print(f"Model parameters saved to {model_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Llama3 Training Script with Accelerate and DeepSpeed")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the model checkpoint and params.json")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed training data file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetuned", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging training loss")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        seed=args.seed,
    )
