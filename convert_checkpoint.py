# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
#
# This script converts a consolidated checkpoint into a sharded checkpoint for
# tensor model parallelism using Fairscale.
import json
import os
import torch
import torch.distributed as dist

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    model_parallel_is_initialized,
)

from model import ModelArgs, Transformer
from tokenizer import Tokenizer

def setup_distributed(model_parallel_size: int):
    """Initializes torch.distributed and Fairscale model parallel groups."""
    dist.init_process_group("nccl")
    initialize_model_parallel(model_parallel_size)
    torch.cuda.set_device(get_model_parallel_rank())

def convert_checkpoint(
    load_path: str,
    output_dir: str,
    params_path: str,
    tokenizer_path: str,
    model_parallel_size: int,
):
    """
    Loads a consolidated checkpoint and saves it into sharded checkpoints
    compatible with the specified model parallel size.
    """
    setup_distributed(model_parallel_size)

    mp_rank = get_model_parallel_rank()
    mp_size = get_model_parallel_world_size()
    print("mp_rank ",mp_rank)
    print("mp_size ", mp_size)
    if mp_rank == 0:
        print(f"Starting checkpoint conversion for model_parallel_size = {mp_size}")
        print(f"Loading consolidated checkpoint from {load_path}...")

    # Each rank loads the full checkpoint to CPU. This is memory-intensive but simple.
    full_state_dict = torch.load(load_path, map_location="cpu")

    # Load model arguments and tokenizer to instantiate the model
    with open(params_path, "r") as f:
        params = json.load(f)
    model_args = ModelArgs(**params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    # Instantiate the sharded model structure.
    # Fairscale will automatically create parallel layers with sharded weights.
    model = Transformer(model_args)
    sharded_state_dict = model.state_dict()

    if mp_rank == 0:
        print("Mapping and sharding weights...")

    # This mapping is based on the Llama 3 architecture and Fairscale's sharding logic.
    # - ColumnParallelLinear: Shards on dimension 0 (output features)
    # - RowParallelLinear: Shards on dimension 1 (input features)
    # - VocabParallelEmbedding: Shards on dimension 0 (vocab size)

    for key, sharded_param in sharded_state_dict.items():
        full_param = full_state_dict[key]
        
        # Identify how the parameter is sharded by its name
        is_column_parallel = any(k in key for k in ["tok_embeddings.weight", "output.weight", "feed_forward.w1", "feed_forward.w3", "attention.wq", "attention.wk", "attention.wv"])
        is_row_parallel = any(k in key for k in ["feed_forward.w2", "attention.wo"])

        if is_column_parallel:
            print("I am is_column_parallel")
            # Shard along dimension 0
            partition_dim = 0
            dim_size = full_param.size(partition_dim)
            if dim_size != sharded_param.size(partition_dim) * mp_size:
                raise ValueError(f"Size mismatch on {key}: ckpt: {dim_size}, model: {sharded_param.size(partition_dim) * mp_size}")

            shard_size = sharded_param.size(partition_dim)
            start = mp_rank * shard_size
            end = start + shard_size
            
            sharded_state_dict[key].copy_(full_param.narrow(partition_dim, start, end))

        elif is_row_parallel:
            print("I am is_row_parallel")
            # Shard along dimension 1
            partition_dim = 1
            dim_size = full_param.size(partition_dim)
            if dim_size != sharded_param.size(partition_dim) * mp_size:
                raise ValueError(f"Size mismatch on {key}: ckpt: {dim_size}, model: {sharded_param.size(partition_dim) * mp_size}")
            
            shard_size = sharded_param.size(partition_dim)
            start = mp_rank * shard_size
            end = start + shard_size
            
            sharded_state_dict[key].copy_(full_param.narrow(partition_dim, start, end))
            
        else:
            # Not a parallel parameter, just copy it
            sharded_state_dict[key].copy_(full_param)

    # Save the sharded checkpoint for the current rank
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"mp_rank_{mp_rank:02d}.pth")
    
    print(f"Saving shard for rank {mp_rank} to {shard_path}...")
    torch.save(sharded_state_dict, shard_path)
    
    dist.barrier()
    if mp_rank == 0:
        print("Checkpoint conversion complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a consolidated Llama 3 checkpoint to a sharded model-parallel checkpoint.")
    
    parser.add_argument("--load_path", type=str, required=True, help="Path to the consolidated checkpoint file (e.g., 'Meta-Llama-3-8B-Instruct/consolidated.00.pth').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the sharded checkpoint files.")
    parser.add_argument("--params_path", type=str, required=True, help="Path to the params.json file from the model directory.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.model file.")
    parser.add_argument("--model_parallel_size", type=int, default=2, help="The number of shards to create (model parallel world size).")
    
    args = parser.parse_args()
    
    convert_checkpoint(
        load_path=args.load_path,
        output_dir=args.output_dir,
        params_path=args.params_path,
        tokenizer_path=args.tokenizer_path,
        model_parallel_size=args.model_parallel_size,
    )
