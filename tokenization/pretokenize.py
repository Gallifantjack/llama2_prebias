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

from metadata.evaluators import evaluate_textual_metrics
from metadata.checkpoint_metadata import expected_stdout
from train_tok.tokenizer import Tokenizer
import polars as pl
import pandas as pd
from torch.utils.data import Sampler

from utils.paths import DATA_CACHE_DIR
from utils.functions import create_global_id, get_tokenizer_model_path

# -----------------------------------------------------------------------------
# Tokenization functions


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    with open(shard, "r") as f:
        data = json.load(f)

    shard_data = []

    for idx, example in tqdm(enumerate(data), position=shard_id):
        global_id = create_global_id(shard_id, idx)
        tokens = enc.encode(example["story"].strip(), bos=True, eos=False)

        # Drop texts with under 5 tokens
        if len(tokens) < 5:
            continue

        token_length = len(tokens)
        shard_data.append((global_id, tokens))

    return shard_data


def pretokenize(vocab_size, debug=False):
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # If debug is True, only take the first 2 shard files
    if debug:
        shard_filenames = shard_filenames[:2]

    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    os.makedirs(bin_dir, exist_ok=True)

    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        all_shard_data = list(executor.map(fun, enumerate(shard_filenames)))

    merged_tokenized_filename = os.path.join(bin_dir, "merged_data.bin")
    merged_idx_filename = os.path.join(bin_dir, "merged_data.idx")

    with open(merged_tokenized_filename, "wb") as f, open(
        merged_idx_filename, "w"
    ) as idx_file:
        for shard_data in all_shard_data:
            for global_id, tokens in shard_data:
                token_length = len(tokens)
                idx_file.write(f"{global_id},{f.tell()},{token_length}\n")
                f.write(np.array(tokens, dtype=np.uint16).tobytes())

    print("Done.")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        type=str,
        choices=["pretokenize"],
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
    if args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size, debug=args.debug)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
