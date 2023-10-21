import os
import random
import subprocess
from shutil import copyfile, rmtree
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import polars as pl

from tinystories import PretokDataset

from visualize_all import visualize_all

from eval import run_evaluation, tokenizer_path, metrics_csv_path

import os
import subprocess


def run_train_script(script_name, args=None):
    # Clear certain environment variables
    for env_var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        os.environ.pop(env_var, None)

    # Construct the base command
    command = ["python3", script_name]

    if args:
        command.extend(args)

    # Run the command
    subprocess.run(command)


if __name__ == "__main__":
    # Different combinations of arguments
    argument_combinations = [
        [
            "--batch_selection",
            "sen_len",
            "--max_iters",
            "200",
            "--out_dir",
            "out/sen_len",
        ],
        [
            "--batch_selection",
            "random",
            "--max_iters",
            "200",
            "--out_dir",
            "out/random",
        ],
    ]

    for args in argument_combinations:
        run_train_script("train.py", args)
