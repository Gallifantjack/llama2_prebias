import os
import random
import subprocess
from shutil import copyfile, rmtree
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import polars as pl

from visualize_all import visualize_all

# from eval import run_evaluation, tokenizer_path, metrics_csv_path

# Constants
BASE_DIR = "data/durability_experiment"
DATASET_DIR = "data/TinyStories_all_data"
SAMPLED_DIR = os.path.join(BASE_DIR, "sampled_batches", "original_order")
SHUFFLED_DIR = os.path.join(BASE_DIR, "sampled_batches", "shuffled_order")


def run_train_script(script_name, additional_args=None):
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        script_name,
        "--eval_interval",
        "1000",
        "--eval_iters",
        "100",
        "--max_iters",
        "10000",
    ]
    if additional_args:
        command.extend(additional_args)
    subprocess.run(command)


def copy_sampled_batches(dataset_dir, batch_files, destination_dir):
    for batch_file in batch_files:
        copyfile(
            os.path.join(dataset_dir, batch_file),
            os.path.join(destination_dir, batch_file),
        )


def training_loop(num_shuffles=3):
    batch_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".bin")]

    # Sample random batches
    num_samples = random.randint(1, len(batch_files))
    print(f"Sampling {num_samples} batches from {len(batch_files)} batches")
    sampled_batches = random.sample(batch_files, num_samples)

    # Create or check directories
    os.makedirs(SAMPLED_DIR, exist_ok=True)
    os.makedirs(SHUFFLED_DIR, exist_ok=True)

    copy_sampled_batches(DATASET_DIR, sampled_batches, SAMPLED_DIR)

    # Train and evaluate on original order
    run_train_script(
        "train.py", ["--data_dir", SAMPLED_DIR, "--out_dir", "models/original_order"]
    )

    # run_evaluation("models/original_order/ckpt/", tokenizer_path, metrics_csv_path)
    # visualize_all("models/original_order/ckpt/")

    # # Train and evaluate on shuffled orders
    # for current_shuffle in range(num_shuffles):
    #     rmtree(SHUFFLED_DIR)
    #     os.makedirs(SHUFFLED_DIR, exist_ok=True)

    #     random.shuffle(sampled_batches)
    #     copy_sampled_batches(DATASET_DIR, sampled_batches, SHUFFLED_DIR)

    #     shuffle_output_dir = f"models/shuffled_order_{current_shuffle+1}"
    #     run_train_script(
    #         "train.py", ["--data_dir", SHUFFLED_DIR, "--out_dir", shuffle_output_dir]
    #     )
    #     run_evaluation(shuffle_output_dir, tokenizer_path, metrics_csv_path)
    #     visualize_all(shuffle_output_dir)

    #     print(f"Shuffle {current_shuffle + 1} done")


if __name__ == "__main__":
    training_loop(num_shuffles=3)


#### THIS RUNS #####
# Goal now to get train to run of terminal above
# get predefined to work remote
# get all of above to run
# get function to return specifc indices based on characteristics sorted

# Assuming you have the following parameters
split = "train"  # or whatever split you're working with (e.g., "test", "valid")
max_seq_len = 1024  # your sequence length
vocab_size = 32000  # or any other value
vocab_source = "llama2"  # or "llama2"

# Create a PretokDataset object
dataset_obj = PretokDataset(
    split=split,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
)


# Function to extract the global indices
def extract_global_indices(dataset, n=None):
    """
    Extracts up to n global indices from the given PretokDataset object.
    If n is None, extracts all indices.

    Args:
    - dataset (PretokDataset): The dataset object from which global indices are to be extracted.
    - n (int, optional): Number of global indices to extract.

    Returns:
    - List[int]: Extracted global indices.
    """
    if n is None:
        return dataset.order
    return dataset.order[:n]


# Now, extract the global indices
desired_length = 100  # For example
indices = extract_global_indices(dataset_obj, desired_length)
print("here is a list of global indices")
print(indices)

train_model(model, optimizer, iter_batches, args, predefined_order=indices)
