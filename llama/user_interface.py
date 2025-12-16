# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import sys
from typing import List, Optional

import fire

from llama import Dialog, Llama

# torchrun --nproc_per_node 1 llama/user_interface.py
def main(
    ckpt_dir: str = "Meta-Llama-3-8B-Instruct/",
    tokenizer_path: str = "Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    An interactive chat interface for the Llama 3 model.

    This function loads a finetuned Llama 3 model and creates a conversational
    loop, allowing a user to chat with the model through the command line.

    Args:
        ckpt_dir (str): The directory containing the model checkpoint.
        tokenizer_path (str): The path to the model tokenizer.
        temperature (float): The temperature for sampling, controlling randomness.
        top_p (float): The nucleus sampling probability.
        max_seq_len (int): The maximum sequence length for the model.
        max_batch_size (int): The maximum batch size for inference.
        max_gen_len (Optional[int]): The maximum number of tokens to generate.
    """
    print("Loading Llama 3 model...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Model loaded successfully.")

    dialog: List[Dialog] = [
        [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please answer the user's questions.",
            }
        ]
    ]

    print("--- Llama 3 Chat Interface ---")
    print('Type "quit" or "exit" to end the conversation.')

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            # Add user input to the dialog
            dialog[0].append({"role": "user", "content": user_input})

            # Generate a response
            result = generator.chat_completion(
                dialog,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )[0]

            response = result["generation"]

            # Print the assistant's response
            print(f"Llama: {response['content']}")

            # Add assistant's response to the dialog for context
            dialog[0].append(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    fire.Fire(main)
