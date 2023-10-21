"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
from torch.utils.data import Sampler


DATA_CACHE_DIR = "data"
MAX_EXAMPLES_PER_SHARD = 1e20


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def create_global_idx(shard_id, example_id):
    return shard_id * MAX_EXAMPLES_PER_SHARD + example_id


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


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []
    batch_metrics = []

    for idx, example in enumerate(tqdm(data, position=shard_id)):
        text = example["story"]
        batch_metric = evaluate_textual_metrics(text, expected_stdout)
        global_idx = create_global_idx(shard_id, idx)
        batch_metric["global_idx"] = global_idx
        batch_metrics.append(batch_metric)
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    if vocab_size == 0:
        tokenized_filename = shard.replace(".json", ".bin")
        metrics_filename = shard.replace(".json", "_metrics.csv")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

        # Save CSV files into a new metrics{N} directory
        metrics_dir = os.path.join(DATA_CACHE_DIR, f"metrics{vocab_size}")
        os.makedirs(metrics_dir, exist_ok=True)  # Ensure directory exists
        shard_basename = os.path.basename(shard)
        metrics_basename = shard_basename.replace(".json", "_metrics.csv")
        metrics_filename = os.path.join(metrics_dir, metrics_basename)

    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

    df_new = pl.DataFrame(batch_metrics)
    df_new.write_csv(metrics_filename)
    print(f"Saved {metrics_filename}")


def merge_csvs(vocab_size, data_dir):
    if vocab_size == 0:
        metrics_dir = data_dir
    else:
        metrics_dir = os.path.join(data_dir, f"metrics{vocab_size}")

    # searching xx path
    print(f"Searching {metrics_dir} for metrics files...")
    metrics_files = glob.glob(os.path.join(metrics_dir, "*_metrics.csv"))

    # Filter out the batch_metrics.csv if it exists in the list
    metrics_files = [f for f in metrics_files if not f.endswith("batch_metrics.csv")]

    combined_df = pl.concat([pl.read_csv(f) for f in metrics_files])
    combined_df.write_csv(os.path.join(data_dir, "batch_metrics.csv"))


def pretokenize(vocab_size):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    # alternatively, process shards sequentially
    # for shard_id, shard in enumerate(shard_filenames):
    #     process_shard((shard_id, shard), vocab_size)

    merge_csvs(vocab_size, data_dir)
    print("Done.")


class PretokDataset(torch.utils.data.Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, select_func=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.select_func = select_func

        # Initialize shard_filenames
        if self.vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        elif self.vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
        else:
            raise ValueError(f"Unknown vocab_source: {self.vocab_source}")
        self.shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        self.shard_filenames = (
            self.shard_filenames[1:]
            if self.split == "train"
            else self.shard_filenames[:1]
        )
        assert len(self.shard_filenames) > 0, f"No bin files found in {bin_dir}"

        # Load metadata
        if self.vocab_source == "llama2":
            metrics_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        elif self.vocab_source == "custom":
            metrics_dir = os.path.join(DATA_CACHE_DIR, f"metrics{self.vocab_size}")
        else:
            raise ValueError(f"Unknown vocab_source: {self.vocab_source}")

        self.metadata_df = pl.read_csv(os.path.join(metrics_dir, "batch_metrics.csv"))
        assert (
            "global_idx" in self.metadata_df.columns
        ), "The metadata does not contain global_idx."

        self.order = self.generate_order(self.shard_filenames)
        if not self.select_func:
            random.shuffle(self.order)
        else:
            selected_indices = self.select_func(self.metadata_df)
            print(f"Selected {len(selected_indices)} indices.")
            selected_indices_set = set(selected_indices)
            self.order = [
                global_ix
                for global_ix in self.order
                if global_ix in selected_indices_set
            ]

            print(f"Filtered order to {len(self.order)} indices.")

    def __len__(self):
        return len(self.order)

    def generate_order(self, shard_filenames):
        global_indices = []

        for shard_id, shard in enumerate(shard_filenames):
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1  # drop the last partial batch
            for ix in range(num_batches):
                global_indices.append(create_global_idx(shard_id, ix))

        return global_indices

    def __getitem__(self, index):
        global_ix = self.order[index]
        shard_id = int(global_ix // MAX_EXAMPLES_PER_SHARD)
        ix = int(global_ix) % (MAX_EXAMPLES_PER_SHARD)
        shard = self.shard_filenames[shard_id]

        m = np.memmap(shard, dtype=np.uint16, mode="r")

        start = int(ix * self.max_seq_len)
        end = int(start + self.max_seq_len + 1)

        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]

        # Fetch metadata for the current global_ix
        filtered_df = self.metadata_df.filter(
            self.metadata_df["global_idx"] == global_ix
        )
        metadata_row = filtered_df.head(1)
        metadata = {
            key: value.item() if isinstance(value, pl.series.series.Series) else value
            for key, value in metadata_row.to_dict().items()
        }

        return x, y, global_ix, metadata


def select_batches_from_sen_len(metadata):
    # Sort by sentence length in descending order
    metadata = metadata.sort("sentence_length", descending=True)

    # Get the starting index for the top 50% of data
    top_50_start_index = len(metadata) // 2

    # Get global indices for the top 50%
    top_50_indices = metadata["global_idx"][top_50_start_index:].to_list()

    # Print the number of selected indices
    print(
        f"Selected {len(top_50_indices)} indices from top 50% based on sentence length."
    )

    return top_50_indices


# -----------------------------------------------------------------------------
# public interface functions


def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")


class Task:
    @staticmethod
    def iter_batches(
        split, batch_size, device, num_workers=0, select_func=None, **dataset_kwargs
    ):
        ds = PretokDataset(split=split, select_func=select_func, **dataset_kwargs)

        # If select_func is provided, don't shuffle. Otherwise, shuffle the data.
        shuffle_data = select_func is None

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle_data,
            pin_memory=True,
            num_workers=num_workers,
        )
        print(f"iter batches stage: {device}")

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y, global_ix, metadata


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", type=str, choices=["download", "pretokenize", "train_vocab"]
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=0,
        help="pretokenization vocab size. 0 = use Llama 2 tokenizer.",
    )
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
