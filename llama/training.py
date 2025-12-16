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
    gradient_accumulation_steps: int = 1,
):
    """
    A simple training script for the Llama3 model.
    """
    # ---- 1. Load Model and Tokenizer ----
    start_time = time.time()
    
    # Initialize accelerator with bf16 mixed precision
    accelerator = Accelerator(mixed_precision='bf16')

    # Ensure the checkpoint directory exists
    if not os.path.isdir(ckpt_dir):
        print(f"Checkpoint directory {ckpt_dir} not found.")
        return

    # Load model parameters
    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    model_args = ModelArgs(**params)

    # Load tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_path)
    # Ensure pad_id is set and within vocab size, using eos_id is a safe default
    tokenizer.pad_id = tokenizer.eos_id
    model_args.vocab_size = tokenizer.n_words
    
    # Instantiate the model
    model = Transformer(model_args)
    # Load the checkpoint if it exists
    checkpoint_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    if checkpoint_paths:
        checkpoint_path = checkpoint_paths[0]
        print(f"Info: Loading checkpoint from {checkpoint_path} for fine-tuning.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Info: No checkpoint file (.pth) found. Initializing model with random weights for training from scratch.")

    print(f"Model and tokenizer loaded in {time.time() - start_time:.2f}s")

    # ---- 2. Prepare Optimizer and Data ----
    optimizer = AdamW8bit(model.parameters(), lr=learning_rate)

    print("Loading training data...")                                                                                        
    with open(data_path, "r", encoding="utf-8") as f:                                                                                                                                              
        train_data = f.read().strip().split('\n\n')
    print(f"Loaded {len(train_data)} training entries.")                                                                     
    
    # Prepare model, optimizer, and a dummy dataloader for accelerator
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
         
    # ---- 3. Training Loop ----
    model.train() # Set the model to training mode

    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        total_loss = 0
        
        # Simple batching
        for i in range(0, len(train_data), batch_size):
            batch_texts = train_data[i:i+batch_size]
            
            # Tokenize and prepare batch
            batch_tokens = [tokenizer.encode(text, bos=True, eos=True) for text in batch_texts]
            
            # Truncate sequences that are longer than max_seq_len
            for j in range(len(batch_tokens)):
                if len(batch_tokens[j]) > model_args.max_seq_len:
                    batch_tokens[j] = batch_tokens[j][:model_args.max_seq_len]

            # Find the max length in the batch for padding
            max_len = max(len(t) for t in batch_tokens)
            
            # Pad and create tensors
            padded_tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in batch_tokens]
            tokens = torch.tensor(padded_tokens, dtype=torch.long)
            
            # Prepare inputs and targets
            # The model should predict the next token, so targets are shifted inputs
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            with accelerator.accumulate(model):
                # Forward pass
                logits = model(inputs, start_pos=0)
                
                # Calculate loss
                loss=F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()

            if (i // batch_size) % log_interval == 0:
                print(f"Batch {i//batch_size + 1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss / (len(train_data)/batch_size):.4f}")


    # ---- 4. Save Model and Optimizer ----
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    model_save_path = os.path.join(output_dir, "llama3_finetuned_model.pth")
    optimizer_save_path = os.path.join(output_dir, "llama3_finetuned_optimizer.pth")
    
    # Unwrap the model before saving
    unwrapped_model = accelerator.unwrap_model(model)
    # Save model state dictionary
    torch.save(unwrapped_model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")

    # Save optimizer state dictionary
    torch.save(optimizer.state_dict(), optimizer_save_path)
    print(f"Optimizer state saved to {optimizer_save_path}")
