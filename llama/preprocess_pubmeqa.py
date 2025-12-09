
import json
from pathlib import Path
import argparse


# python3 llama/preprocess_pubmeqa.py --input_dir pubmedqa --output_file pubmedqa/pubmeqa_preprocessed_sj.txt
def format_entry(entry):
    """Formats a single JSON entry into a question-answer string."""
    question = entry.get("QUESTION", "").strip()
    # The answer is in LONG_ANSWER. CONTEXTS are just for context.
    answer = entry.get("LONG_ANSWER", "").strip()

    if not answer:
        # Fallback to contexts if LONG_ANSWER is missing
        contexts = entry.get("CONTEXTS", [])
        answer = " ".join(contexts).strip()

    if question and answer:
        return f"Question: {question}\nAnswer: {answer}"
    return None

def process_json_file(file_path):
    """Processes a single JSON file and returns a list of formatted strings."""
    formatted_entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            for key, value in data.items():
                formatted_entry = format_entry(value)
                if formatted_entry:
                    formatted_entries.append(formatted_entry)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
    return formatted_entries

def main():
    parser = argparse.ArgumentParser(description="Preprocess PubMEQA JSON dataset for LLaMA training.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing the PubMEQA JSON files (e.g., ../pubmeqa).")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to the output file to save preprocessed data (e.g., pubmeqa_preprocessed.txt).")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}")
        return

    # Use rglob to find all json files recursively
    json_files = list(input_dir.rglob("*.json"))
    
    if not json_files:
        print(f"Warning: No .json files found in {input_dir}")
        return

    train_files = [f for f in json_files if 'train' in f.name]
    test_files = [f for f in json_files if 'test' in f.name]

    print(f"Found {len(json_files)} total JSON files.")
    print(f"Found {len(train_files)} training files.")
    print(f"Found {len(test_files)} test files.")

    # Process training files
    train_entries = []
    print("\nProcessing training files...")
    for file_path in train_files:
        print(f"  - {file_path.name}")
        train_entries.extend(process_json_file(file_path))
    
    # Process test files
    test_entries = []
    print("\nProcessing test files...")
    for file_path in test_files:
        print(f"  - {file_path.name}")
        test_entries.extend(process_json_file(file_path))

    # Derive output file paths
    output_train_file = output_file.with_suffix('').as_posix() + "_train.txt"
    output_test_file = output_file.with_suffix('').as_posix() + "_test.txt"

    # Write training data
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(entry + "\n\n")

    # Write test data
    with open(output_test_file, 'w', encoding='utf-8') as f:
        for entry in test_entries:
            f.write(entry + "\n\n")

    print(f"\nPreprocessing complete.")
    print(f"Processed {len(train_entries)} training entries, saved to {output_train_file}")
    print(f"Processed {len(test_entries)} test entries, saved to {output_test_file}")

if __name__ == "__main__":
    main()
