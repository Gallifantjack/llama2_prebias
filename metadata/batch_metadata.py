# -----------------------------------------------------------------------------

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import itertools

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

import pandas as pd
import dask.dataframe as dd

from metadata.evaluators import evaluate_textual_metrics
from train_tok.tokenizer import Tokenizer
from utils.paths import DATA_CACHE_DIR
from utils.functions import get_tokenizer_model_path

# -----------------------------------------------------------------------------

expected_stdout = b"Once upon a time, there was a little girl named Lily..."


# -----------------------------------------------------------------------------
# Metadata functions
def string_to_list(string_repr):
    try:
        # Trim the brackets and split by comma
        return [int(token) for token in string_repr[1:-1].split()]
    except (ValueError, IndexError):
        return []


def detokenize_from_parquet(parquet_file, tokenizer_path, debug=False, chunk_size=5000):
    enc = Tokenizer(tokenizer_model=tokenizer_path)

    texts = []
    global_ids = []

    # Using Dask to read Parquet file in chunks
    ddf = dd.read_parquet(parquet_file, engine="pyarrow")

    # If debugging, limit the dataframe size
    if debug:
        partitions = ddf.to_delayed()[
            :1
        ]  # Limiting to the first partition for debug mode
    else:
        partitions = ddf.to_delayed()

    for partition in partitions:
        df = partition.compute()

        token_lists = df["tokens"].apply(string_to_list)
        detokenized_texts_chunk = [enc.decode(tokens) for tokens in token_lists]

        global_ids_chunk = df["id"].tolist()

        texts.extend(detokenized_texts_chunk)
        global_ids.extend(global_ids_chunk)
        print(detokenized_texts_chunk[:5])

    return texts, global_ids


def worker(texts, global_indices):
    shard_metrics = []
    for idx, detokenized_text in enumerate(texts):
        metrics = evaluate_textual_metrics(
            detokenized_text, expected_stdout.decode("utf-8")
        )
        metrics["id"] = global_indices[idx]
        shard_metrics.append(metrics)
    return shard_metrics


def compute_metadata(vocab_size, debug=False):
    parquet_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}", "merged_data.parquet"
    )
    tokenizer_path = get_tokenizer_model_path(vocab_size)

    # Detokenize stories
    detokenized_texts, global_indices = detokenize_from_parquet(
        parquet_path, tokenizer_path, debug
    )

    # print the first 10 detokenized texts
    print(detokenized_texts[:10])
    print(global_indices[:10])

    # Prepare an empty list to hold all metadata
    all_metadata = []

    # Parallelizing the computation of metrics
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
    all_metadata = list(itertools.chain(*results))

    # Convert to DataFrame
    df_metadata = pd.DataFrame(all_metadata)

    # Load the original DataFrame
    df_original = pd.read_parquet(parquet_path)

    # Merge the original DataFrame with the metadata on "id"
    df_combined = pd.merge(df_original, df_metadata, on="id", how="left")

    # Save to a new Parquet file for comparison
    new_parquet_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}", "merged_data_with_metadata.parquet"
    )
    df_combined.to_parquet(new_parquet_path, index=False)

    print(f"Processed data for vocab size: {vocab_size}")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["compute_metadata"])
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=0,
        help="pretokenization vocab size. 0 = use Llama 2 tokenizer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode, processes only the first 200 entries in the compute_metadata stage.",
    )
    args = parser.parse_args()

    if args.stage == "compute_metadata":
        compute_metadata(vocab_size=args.vocab_size, debug=args.debug)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
