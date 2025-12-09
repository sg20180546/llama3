# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import time
from pathlib import Path
from typing import List
import re

import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from generation import Llama

@torch.no_grad()
def calculate_perplexity(model: Transformer, tokenizer: Tokenizer, text: str, device: str) -> float:
    """
    Calculates the perplexity of a given text under the model.
    Lower is better.
    """
    model.eval() # Set the model to evaluation mode
    
    tokens = tokenizer.encode(text, bos=True, eos=True)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # A sequence length of 1 can't be evaluated
    if tokens.shape[1] <= 1:
        return float('nan')

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits = model(inputs, start_pos=0)
    
    # Calculate loss using cross_entropy
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_id)
    
    # Perplexity is the exponential of the loss
    perplexity = torch.exp(loss).item()
    
    return perplexity

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    test_data_path: str,
    max_gen_len: int = 64, # Shorter for decision generation
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    """
    This script evaluates a fine-tuned Llama3 model on a preprocessed test dataset.
    It calculates and reports two metrics:
    1. Perplexity (PPL): How well the model predicts the ground truth long answer.
    2. Final Decision Accuracy: Whether the model's generated answer contains the correct final decision ('yes', 'no', 'maybe').
    """
    # ---- 1. Load Model for Evaluation ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Re-build model for PPL calculation
    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    model_args = ModelArgs(**params)
    
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    
    model = Transformer(model_args).to(device)
    
    checkpoint_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
    print(f"Loading checkpoint from {checkpoint_path} for evaluation.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    # Also build the generator for accuracy testing
    generator = Llama.build(
        ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=1024, max_batch_size=2
    )

    # ---- 2. Load Test Data ----
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = f.read().strip().split('\n\n')
    print(f"Loaded {len(test_data)} test entries.")

    total_perplexity = 0
    correct_decisions = 0
    processed_entries = 0
    
    # ---- 3. Evaluation Loop ----
    for i, entry in enumerate(test_data):
        # Parse entry
        try:
            question = re.search(r"Question: (.*?)\nAnswer:", entry, re.DOTALL).group(1).strip()
            ground_truth_answer = re.search(r"Answer: (.*?)\nFinal Decision:", entry, re.DOTALL).group(1).strip()
            final_decision = re.search(r"Final Decision: (.*)", entry).group(1).strip()
        except AttributeError:
            print(f"Warning: Skipping malformed entry #{i+1}: {entry[:100]}...")
            continue
            
        # --- Metric 1: Perplexity ---
        ppl = calculate_perplexity(model, tokenizer, ground_truth_answer, device)
        if not (ppl != ppl): # check for NaN
            total_perplexity += ppl
        
        # --- Metric 2: Final Decision Accuracy ---
        prompt = f"Question: {question}\nAnswer:"
        generated_result = generator.text_completion(
            [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
        )[0]
        generated_answer = generated_result['generation'].strip().lower()

        # Check if the generated answer starts with the correct decision
        if generated_answer.startswith(final_decision):
            correct_decisions += 1
        
        processed_entries += 1

        print(f"--- Entry {i+1}/{len(test_data)} ---")
        print(f"  PPL: {ppl:.4f}")
        print(f"  Correct Decision: '{final_decision}' -> Generated starts with '{final_decision}': {generated_answer.startswith(final_decision)}")
        print(f"  Generated Answer Snippet: {generated_answer[:100]}...")


    # ---- 4. Report Final Metrics ----
    average_perplexity = total_perplexity / processed_entries if processed_entries > 0 else 0
    accuracy = (correct_decisions / processed_entries) * 100 if processed_entries > 0 else 0

    print("\n\n--- Evaluation Complete ---")
    print(f"Successfully processed {processed_entries} entries.")
    print(f"\nAverage Perplexity (lower is better): {average_perplexity:.4f}")
    print(f"Final Decision Accuracy: {accuracy:.2f}% ({correct_decisions}/{processed_entries})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama3 model.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the fine-tuned model checkpoint and params.json.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the preprocessed test data file.")
    
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        test_data_path=args.test_data_path,
    )
