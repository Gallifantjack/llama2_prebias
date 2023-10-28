import os
import random
import subprocess
from shutil import copyfile, rmtree
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        # flesch_kincaid_grade - Ascending
        [
            "--transform_method",
            "sort_ascending",
            "--transform_column",
            "flesch_kincaid_grade",
            "--max_iters",
            "1000",
            "--eval_interval",
            "100",
            "--eval_iters",
            "10",
            "--out_dir",
            "out/flesch_kincaid_asc",
        ],
        # flesch_kincaid_grade - Descending
        [
            "--transform_method",
            "sort_descending",
            "--transform_column",
            "flesch_kincaid_grade",
            # "--max_iters",
            # "1000",
            # "--eval_interval",
            # "100",
            # "--eval_iters",
            # "10",
            "--out_dir",
            "out/flesch_kincaid_desc",
        ]
        # subjectivity_score ascending
        [
            "--transform_method",
            "sort_ascending",
            "--transform_column",
            "subjectivity_score",
            # "--max_iters",
            # "10000",
            # "--eval_interval",
            # "1000",
            # "--eval_iters",
            # "100",
            "--out_dir",
            "out/subj_score_asc",
        ],
        # subjectivity_score descending
        [
            "--transform_method",
            "sort_descending",
            "--transform_column",
            "subjectivity_score",
            # "--max_iters",
            # "10000",
            # "--eval_interval",
            # "1000",
            # "--eval_iters",
            # "100",
            "--out_dir",
            "out/subj_score_desc",
        ],
    ]

for args in argument_combinations:
    run_train_script("modelling/train.py", args)