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
        # Sentence length - Ascending
        [
            "--batch_selection",
            "sort_column",
            "--sort_by_column",
            "flesch_kincaid_grade",
            "--sort_by_direction",
            "asc",
            "--max_iters",
            "10000",
            "--eval_interval",
            "1000",
            "--eval_iters",
            "100",
            "--out_dir",
            "out/flesch_kincaid_asc",
        ],
        # Sentence length - Descending
        [
            "--batch_selection",
            "sort_column",
            "--sort_by_column",
            "flesch_kincaid_grade",
            "--sort_by_direction",
            "desc",
            "--max_iters",
            "10000",
            "--eval_interval",
            "1000",
            "--eval_iters",
            "100",
            "--out_dir",
            "out/flesch_kincaid_desc",
        ],
        # subjectivity_score
        [
            "--batch_selection",
            "sort_column",
            "--sort_by_column",
            "subjectivity_score",
            "--sort_by_direction",
            "asc",
            "--max_iters",
            "10000",
            "--eval_interval",
            "1000",
            "--eval_iters",
            "100",
            "--out_dir",
            "out/subj_score_asc",
        ],
        # subjectivity_score
        [
            "--batch_selection",
            "sort_column",
            "--sort_by_column",
            "subjectivity_score",
            "--sort_by_direction",
            "desc",
            "--max_iters",
            "10000",
            "--eval_interval",
            "1000",
            "--eval_iters",
            "100",
            "--out_dir",
            "out/subj_score_desc",
        ],
    ]

for args in argument_combinations:
    run_train_script("train.py", args)

    for args in argument_combinations:
        run_train_script("train.py", args)
