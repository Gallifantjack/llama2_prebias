import os
from collections import defaultdict

# Base paths
base_path = "/home/legionjgally/Desktop/mit/llama2_prebias/out"


# Ensure the directory exists
def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)


# Utility function for nested defaultdicts
def nested_defaultdict():
    return defaultdict(nested_defaultdict)


# Function to build checkpoint paths
def get_checkpoint_path(model_name, epoch):
    if model_name == "ckpt":
        return os.path.join(base_path, "ckpt", f"ckpt_{epoch}.pt")
    return os.path.join(base_path, model_name, "ckpt", f"ckpt_{epoch}.pt")


# Function to build checkpoint paths
def checkpoint_output_results_path(model_name):
    return os.path.join(
        base_path, model_name, "metadata", "checkpoint_output_results.parquet"
    )


# Function to build batch results paths
def batch_parquet_file_path(model_name):
    return os.path.join(base_path, model_name, "metadata", "batch_results.parquet")
