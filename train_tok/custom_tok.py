import argparse
import glob
import math
import json
import os
import random
import itertools
from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import glob
import cProfile

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from evaluators import evaluate_textual_metrics
from eval import expected_stdout
from tokenizer import Tokenizer
import polars as pl
import pandas as pd


DATA_CACHE_DIR = "data"


def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


# -----------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    """
    To tokenize data with a custom tokenizer we train ourselves with sentencepiece
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        type=str,
        choices=["train_vocab"],
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=0,
        help="pretokenization vocab size. 0 = use Llama 2 tokenizer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",  # This will store 'True' if the --debug flag is used, otherwise it defaults to 'False'
        help="Use debug mode, processes only the first 2 shard files in the pretokenize stage.",
    )
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
