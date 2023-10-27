import argparse
import glob
import math
import json
import os
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import cProfile
import itertools

import numpy as np
import sentencepiece as spm
import torch.distributed as dist
from tqdm import tqdm

import polars as pl
import pandas as pd
from torch.utils.data import Sampler

# relative imports
from metadata.evaluators import evaluate_textual_metrics
from train_tok.tokenizer import Tokenizer

from utils.paths import DATA_CACHE_DIR
from utils.functions import get_tokenizer_model_path

# -----------------------------------------------------------------------------
expected_stdout = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"


# -----------------------------------------------------------------------------
# Metadata functions
def detokenize_from_bin(bin_file, idx_file, tokenizer_path, debug=False):
    CHUNK_SIZE = 10 * 1024 * 1024  # For example, 10 MB
    enc = Tokenizer(tokenizer_model=tokenizer_path)
    texts, global_ids, empty_indices = [], [], []
    offsets = []

    with open(idx_file, "r") as idx:
        for line in idx:
            global_id, byte_offset, token_length = map(int, line.strip().split(","))
            offsets.append((global_id, byte_offset, token_length))
            if debug and len(offsets) >= 200:
                break

    with open(bin_file, "rb") as f:
        chunk_start = 0
        while chunk_start < os.path.getsize(bin_file):
            f.seek(chunk_start)
            chunk = f.read(CHUNK_SIZE)
            next_chunk_start = chunk_start + CHUNK_SIZE

            # Process offsets that are inside the current chunk
            while offsets and offsets[0][1] < next_chunk_start:
                global_id, byte_offset, token_length = offsets.pop(0)
                relative_offset = byte_offset - chunk_start
                tokens = np.frombuffer(
                    chunk[relative_offset : relative_offset + token_length * 2],
                    dtype=np.uint16,
                )
                decoded_text = enc.decode(tokens.tolist())
                if not decoded_text.strip():
                    empty_indices.append(global_id)
                texts.append(decoded_text)
                global_ids.append(global_id)

            chunk_start = next_chunk_start

    return texts, global_ids, empty_indices


def worker(texts, global_indices):
    shard_metrics = []
    for idx, detokenized_text in enumerate(texts):
        metrics = evaluate_textual_metrics(
            detokenized_text, expected_stdout.decode("utf-8")
        )
        metrics["global_id"] = global_indices[idx]
        shard_metrics.append(metrics)
    return shard_metrics


def compute_shard_metrics(bin_file, tokenizer_path, vocab_size):
    idx_file = bin_file.replace(".bin", ".idx")
    detokenized_texts, global_indices, empty_global_ids = detokenize_from_bin(
        bin_file, idx_file, tokenizer_path
    )

    # Parallelizing the computation
    with ProcessPoolExecutor() as executor:
        chunk_size = math.ceil(len(detokenized_texts) / os.cpu_count())
        futures = [
            executor.submit(
                worker,
                detokenized_texts[i : i + chunk_size],
                global_indices[i : i + chunk_size],
            )
            for i in range(0, len(detokenized_texts), chunk_size)
        ]
        results = [f.result() for f in futures]

    # Flattening the results
    shard_metrics = list(itertools.chain(*results))

    return shard_metrics, empty_global_ids


def compute_metadata(vocab_size, debug=False):
    bin_file = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}", "merged_data.bin")
    tokenizer_path = get_tokenizer_model_path(vocab_size)

    # Compute metrics and get empty IDs
    shard_metrics, empty_global_ids = compute_shard_metrics(
        bin_file, tokenizer_path, vocab_size
    )

    # Save results to CSV
    metrics_output_path = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}/metadata.csv")
    pl.DataFrame(shard_metrics).write_csv(metrics_output_path)

    empty_ids_output_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}/empty_ids.csv"
    )
    pl.DataFrame({"empty_global_id": empty_global_ids}).write_csv(empty_ids_output_path)

    print(f"Processed data for vocab size: {vocab_size}")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        type=str,
        choices=["compute_metadata"],
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
    if args.stage == "compute_metadata":
        compute_metadata(vocab_size=args.vocab_size, debug=args.debug)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
