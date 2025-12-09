# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

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
):
    """
    A simple training script for the Llama3 model.
    """
    # ---- 1. Load Model and Tokenizer ----
    start_time = time.time()
    
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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = Transformer(model_args).to(device)

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
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Dummy data for demonstration
    # Replace this with your actual data loading logic
    # train_data = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "Llama models are a family of large language models.",
    #     "Training a language model requires a significant amount of data and compute.",
    #     "PyTorch is a popular deep learning framework.",
    # ]│  67 +     # Load the preprocessed training data                                                                                    │
    print("Loading training data...")                                                                                        
    with open(data_path, "r", encoding="utf-8") as f:                                                                                                                                              
        train_data = f.read().strip().split('\n\n')
    print(f"Loaded {len(train_data)} training entries.")                                                                     
         
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
            tokens = torch.tensor(padded_tokens, dtype=torch.long, device=device)
            
            # Prepare inputs and targets
            # The model should predict the next token, so targets are shifted inputs
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            # Forward pass
            logits = model(inputs, start_pos=0)
            
            # Calculate loss
            # Reshape logits and targets for cross_entropy
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tokenizer.pad_id)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

            if (i // batch_size) % log_interval == 0:
                print(f"Batch {i//batch_size + 1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss / (len(train_data)/batch_size):.4f}")


    # ---- 4. Save Model and Optimizer ----
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    model_save_path = os.path.join(output_dir, "llama3_finetuned_model.pth")
    optimizer_save_path = os.path.join(output_dir, "llama3_finetuned_optimizer.pth")
    
    # Save model state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")

    # Save optimizer state dictionary
    torch.save(optimizer.state_dict(), optimizer_save_path)
    print(f"Optimizer state saved to {optimizer_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Llama3 Training Script")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the model checkpoint and params.json")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed training data file.")         
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetuned", help="Directory to save the fine-tuned model and optimizer state")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
