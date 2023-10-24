"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
import itertools
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob

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


# -----------------------------------------------------------------------------
# Tokenization functions
def create_global_idx(shard_id, idx):
    # A simple method to create unique ID by combining shard_id and idx
    # You can modify this function if needed.
    return f"{shard_id}_{idx}"


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    with open(shard, "r") as f:
        data = json.load(f)

    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    shard_basename = os.path.basename(shard)
    bin_basename = shard_basename.replace(".json", ".bin")
    tokenized_filename = os.path.join(bin_dir, bin_basename)
    idx_basename = shard_basename.replace(".json", ".idx")
    idx_filename = os.path.join(bin_dir, idx_basename)

    with open(tokenized_filename, "wb") as f, open(idx_filename, "w") as idx_file:
        for example in tqdm(data, position=shard_id):
            global_idx = create_global_idx(shard_id, data.index(example))
            tokens = enc.encode(example["story"].strip(), bos=True, eos=False)
            token_length = len(tokens)

            # Write tokenized data to binary file
            f.write(np.array(tokens, dtype=np.uint16).tobytes())

            # Write global_idx and token_length to idx file
            idx_file.write(f"{global_idx},{token_length}\n")


def pretokenize(vocab_size):
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    os.makedirs(bin_dir, exist_ok=True)

    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))

    print("Done.")


# -----------------------------------------------------------------------------
# Metadata functions


def detokenize_from_bin(bin_file, idx_file, tokenizer_path):
    enc = Tokenizer(tokenizer_model=tokenizer_path)

    texts = []
    global_ids = []

    with open(bin_file, "rb") as f, open(idx_file, "r") as idx:
        for line in idx:
            global_id, token_length = line.strip().split(",")
            token_length = int(token_length)
            tokens = np.frombuffer(
                f.read(token_length * 2), dtype=np.uint16
            )  # Read 2 bytes for each token

            # Convert numpy array to list before decoding
            decoded_text = enc.decode(tokens.tolist())

            texts.append(decoded_text)
            global_ids.append(global_id)

    return texts, global_ids


def compute_shard_metrics(bin_file, tokenizer_path, vocab_size):
    idx_file = bin_file.replace(".bin", ".idx")
    detokenized_texts, global_indices = detokenize_from_bin(
        bin_file, idx_file, tokenizer_path
    )

    shard_metrics = []
    for idx, detokenized_text in enumerate(detokenized_texts):
        metrics = evaluate_textual_metrics(
            detokenized_text, expected_stdout.decode("utf-8")
        )
        metrics["global_id"] = global_indices[idx]  # Attach global_id to the metrics
        shard_metrics.append(metrics)

    shard_name = os.path.basename(bin_file).replace(".bin", "")
    shard_metrics_output_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}/metadata_for_{shard_name}.csv"
    )
    pl.DataFrame(shard_metrics).write_csv(shard_metrics_output_path)

    return shard_metrics_output_path


def compute_metadata(vocab_size):
    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
    tokenizer_path = get_tokenizer_model_path(vocab_size)

    # Parallelize the processing of each shard
    with ProcessPoolExecutor() as executor:
        shard_metric_files = list(
            executor.map(
                compute_shard_metrics,
                bin_files,
                itertools.repeat(tokenizer_path, len(bin_files)),
                itertools.repeat(vocab_size, len(bin_files)),
            )
        )

    print(f"Processed all shards for vocab size: {vocab_size}")

    # Concatenate the computed shard metrics at the end
    concatenate_shards(vocab_size)


def concatenate_shards(vocab_size):
    # Pattern to match all shard metric files
    pattern = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}/metadata_for_*.csv")
    shard_files = sorted(glob.glob(pattern))

    all_metrics = [pl.read_csv(file) for file in shard_files]
    overall_metrics_df = pl.concat(all_metrics, how="vertical")

    overall_metrics_output_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}/overall_metadata_for_vocab_{vocab_size}.csv"
    )
    overall_metrics_df.write_csv(overall_metrics_output_path)
    print(f"Saved overall metadata results to {overall_metrics_output_path}")


# -----------------------------------------------------------------------------
# Dataset class
class PretokDataset(torch.utils.data.Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, select_func=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

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
            metrics_dir = os.path.join(DATA_CACHE_DIR, "tok0")
        elif self.vocab_source == "custom":
            metrics_dir = os.path.join(DATA_CACHE_DIR, f"metrics{self.vocab_size}")
        else:
            raise ValueError(f"Unknown vocab_source: {self.vocab_source}")

        self.metadata_df = pl.read_csv(
            os.path.join(metrics_dir, "overall_metadata_for_vocab_0.csv")
        )
        assert (
            "global_id" in self.metadata_df.columns
        ), "The metadata does not contain global_idx."

        df_dict = self.metadata_df.to_dict(as_series=False)
        global_idx_values = df_dict.pop(
            "global_id"
        )  # remove global_idx from dict and get its values

        # Construct metadata_dict where global_idx serves as the key
        self.metadata_dict = {
            idx: {key: df_dict[key][i] for key in df_dict}
            for i, idx in enumerate(global_idx_values)
        }

        # Construct the order
        self.order = self._construct_order()

    def _construct_order(self):
        """Construct the order based on shard filenames and other details."""
        return generate_global_order(self.shard_filenames, self.max_seq_len)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        global_ix = self.order[int(index)]

        # Split the string on the underscore
        shard_str, row_str = global_ix.split("_")

        # Convert the shard string and row string to integers
        shard_id = int(shard_str)
        ix = int(row_str)

        shard = self.shard_filenames[shard_id]

        m = np.memmap(shard, dtype=np.uint16, mode="r")

        start = int(ix * self.max_seq_len)
        end = int(start + self.max_seq_len + 1)

        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]

        # Fetch metadata for the current global_ix using the dictionary
        metadata = self.metadata_dict.get(global_ix, {})

        return x, y, global_ix, metadata


class DynamicSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, select_func=None):
        self.dataset = dataset
        self.select_func = select_func
        self.order = self.generate_order()

        if self.select_func:
            selected_indices = self.select_func(dataset.metadata_df)
            print(f"Selected {len(selected_indices)} indices.")
            print(f"First few selected indices: {selected_indices[:10]}")

            selected_indices_set = set(selected_indices)
            self.order = [
                global_ix
                for global_ix in self.order
                if global_ix in selected_indices_set
            ]

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)

    def generate_order(self):
        """Construct the order based on shard filenames and other details."""
        return generate_global_order(
            self.dataset.shard_filenames, self.dataset.max_seq_len
        )


def generate_global_order(shard_filenames, max_seq_len):
    global_indices = []
    for shard_id, shard in enumerate(shard_filenames):
        m = np.memmap(shard, dtype=np.uint16, mode="r")
        num_batches = len(m) // max_seq_len
        num_batches -= 1  # drop the last partial batch
        for ix in range(num_batches):
            global_indices.append(create_global_idx(shard_id, ix))
    return global_indices


def select_batches_sorted_by_column(metadata, column_name, ascending=True):
    # Use polars to sort the metadata by the column
    metadata_sorted = metadata.sort(column_name, descending=not ascending)

    # Get the correct column position for "global_idx"
    global_idx_position = metadata_sorted.columns.index("global_id")

    # Extract the global_idx column values.
    sorted_indices = [row[global_idx_position] for row in metadata_sorted.rows()]

    return sorted_indices


# -----------------------------------------------------------------------------
# public interface functions


class Task:
    @staticmethod
    def iter_batches(
        split, batch_size, device, num_workers=0, select_func=None, **dataset_kwargs
    ):
        ds = PretokDataset(split=split, **dataset_kwargs)

        # If select_func is provided, don't shuffle. Otherwise, shuffle the data.
        sampler = DynamicSampler(ds, select_func=select_func)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
        )
        print(f"iter batches stage: {device}")

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            print(x)
            print(y)
            print(global_ix)
            print(metadata)
            yield x, y, global_ix, metadata


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
        "stage",
        type=str,
        choices=["download", "pretokenize", "train_vocab", "compute_metadata"],
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
    elif args.stage == "compute_metadata":
        compute_metadata(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
