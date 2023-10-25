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

    byte_offset = 0  # Start byte offset

    with open(tokenized_filename, "wb") as f, open(idx_filename, "w") as idx_file:
        for example in tqdm(data, position=shard_id):
            global_idx = create_global_idx(shard_id, data.index(example))
            tokens = enc.encode(example["story"].strip(), bos=True, eos=False)
            token_length = len(tokens)

            # Write global_idx, byte offset and token_length to idx file
            idx_file.write(f"{global_idx},{byte_offset},{token_length}\n")

            # Write tokenized data to binary file
            f.write(np.array(tokens, dtype=np.uint16).tobytes())

            # Update byte offset for the next iteration
            byte_offset += token_length * np.uint16().nbytes


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
    empty_indices = []

    with open(bin_file, "rb") as f, open(idx_file, "r") as idx:
        for line in idx:
            global_id, byte_offset, token_length = line.strip().split(",")
            byte_offset, token_length = int(byte_offset), int(token_length)

            # Use byte offset to jump directly to the data
            f.seek(byte_offset)
            tokens = np.frombuffer(f.read(token_length * 2), dtype=np.uint16)

            # Convert numpy array to list before decoding
            decoded_text = enc.decode(tokens.tolist())

            if not decoded_text.strip():
                empty_indices.append(global_id)

            texts.append(decoded_text)

            global_ids.append(global_id)

    return texts, global_ids, empty_indices


def compute_shard_metrics(bin_file, tokenizer_path, vocab_size):
    idx_file = bin_file.replace(".bin", ".idx")
    detokenized_texts, global_indices, empty_global_ids = detokenize_from_bin(
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

    # Save shard metrics to CSV
    shard_metrics_output_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}/metadata_for_{shard_name}.csv"
    )
    pl.DataFrame(shard_metrics).write_csv(shard_metrics_output_path)

    # Save empty global IDs for this shard
    empty_global_ids_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}/empty_ids_for_{shard_name}.csv"
    )
    pl.DataFrame({"empty_global_id": empty_global_ids}).write_csv(empty_global_ids_path)

    return shard_metrics_output_path, empty_global_ids_path


def compute_metadata(vocab_size):
    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
    tokenizer_path = get_tokenizer_model_path(vocab_size)

    # Parallelize the processing of each shard
    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                compute_shard_metrics,
                bin_files,
                itertools.repeat(tokenizer_path, len(bin_files)),
                itertools.repeat(vocab_size, len(bin_files)),
            )
        )

    # Unpack the results into separate lists
    shard_metric_files, empty_ids_files = zip(*results)

    print(f"Processed all shards for vocab size: {vocab_size}")

    # Concatenate the computed shard metrics and empty ids
    concatenate_shards(vocab_size, shard_metric_files, empty_ids_files)

    # Perform sanity checks
    # Check the contents of empty_ids in the metadata- have we imputed them all
    confirmed_empty_ids, not_empty_ids = verify_empty_texts(vocab_size)
    print(f"Confirmed Empty IDs: {confirmed_empty_ids}")

    empty_id_metadata_df = inspect_metadata_for_empty_ids(
        vocab_size, confirmed_empty_ids
    )
    print(empty_id_metadata_df)

    ## Does every column in the metadata have a value for every row?
    missing_values_global_ids = verify_metadata_values(vocab_size)
    if missing_values_global_ids:
        print(
            f"Global IDs with missing values in metadata: {missing_values_global_ids}"
        )
    else:
        print("All rows in metadata have values in each column!")


def concatenate_shards(vocab_size, shard_metric_files, empty_ids_files):
    # Concatenate metrics
    pattern = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}/metadata_for_*.csv")
    shard_files = sorted(glob.glob(pattern))
    all_metrics = [pl.read_csv(file) for file in shard_files]
    overall_metrics_df = pl.concat(all_metrics, how="vertical")
    overall_metrics_output_path = os.path.join(
        DATA_CACHE_DIR,
        f"tok{vocab_size}/overall_metadata_for_vocab_{vocab_size}.csv",
    )
    overall_metrics_df.write_csv(overall_metrics_output_path)

    # Concatenate empty global ids
    all_empty_ids = [pl.read_csv(file) for file in empty_ids_files]
    overall_empty_ids_df = pl.concat(all_empty_ids, how="vertical")
    overall_empty_ids_output_path = os.path.join(
        DATA_CACHE_DIR,
        f"tok{vocab_size}/overall_empty_ids_for_vocab_{vocab_size}.csv",
    )
    overall_empty_ids_df.write_csv(overall_empty_ids_output_path)

    print(f"Saved overall metadata results to {overall_metrics_output_path}")
    print(f"Saved overall empty global IDs to {overall_empty_ids_output_path}")


# Check scripts
def verify_empty_texts(vocab_size):
    # Load the overall CSV containing empty global IDs
    empty_ids_path = os.path.join(
        DATA_CACHE_DIR,
        f"tok{vocab_size}/overall_empty_ids_for_vocab_{vocab_size}.csv",
    )
    df = pl.read_csv(empty_ids_path)
    empty_global_ids = df["empty_global_id"].to_list()

    confirmed_empty = []
    not_empty = []

    tokenizer_path = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model=tokenizer_path)

    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

    for bin_file in bin_files:
        idx_file = bin_file.replace(".bin", ".idx")
        with open(bin_file, "rb") as f, open(idx_file, "r") as idx:
            for line in idx:
                global_id, byte_offset, token_length = line.strip().split(",")
                byte_offset, token_length = int(byte_offset), int(token_length)

                # If the current global ID is not in our list of empty IDs, skip it
                if global_id not in empty_global_ids:
                    continue

                # Use byte offset to jump directly to the data
                f.seek(byte_offset)
                tokens = np.frombuffer(f.read(token_length * 2), dtype=np.uint16)

                # Convert numpy array to list before decoding
                decoded_text = enc.decode(tokens.tolist())

                if not decoded_text.strip():
                    confirmed_empty.append(global_id)
                else:
                    not_empty.append(global_id)

    return confirmed_empty, not_empty


def inspect_metadata_for_empty_ids(vocab_size, confirmed_empty_ids):
    metadata_path = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}/overall_metadata_for_vocab_{vocab_size}.csv")
    df = pl.read_csv(metadata_path)
    # Filter the metadata dataframe for rows matching the empty IDs
    empty_id_metadata = df.filter(df["global_id"].is_in(confirmed_empty_ids))
    return empty_id_metadata


def verify_metadata_values(vocab_size):
    # Load the overall metadata CSV
    metadata_path = os.path.join(
        DATA_CACHE_DIR,
        f"tok{vocab_size}/overall_metadata_for_vocab_{vocab_size}.csv",
    )
    df = pl.read_csv(metadata_path)

    # Create a mask for rows with any null values
    masks = [
        pl.col(column_name).is_null().alias(column_name) for column_name in df.columns
    ]
    null_masks = df.select(*masks)

    # Check which rows have any null values
    rows_with_missing_values = null_masks.filter(null_masks.sum(axis=1) > 0)[
        "global_id"
    ].to_list()

    return rows_with_missing_values


# -----------------------------------------------------------------------------

class PretokDataset(torch.utils.data.Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, select_func=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

        # Determine the appropriate directory for .bin files
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}" if vocab_source == "custom" else "tok0")
        print(f"Expected .bin file directory: {bin_dir}")
        
        self.shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        if split == "train":
            self.shard_filenames = self.shard_filenames[1:]
        else:
            self.shard_filenames = self.shard_filenames[:1]
        
        assert self.shard_filenames, f"No bin files found in {bin_dir}"
        print(f"Number of .bin files found: {len(self.shard_filenames)}")

        # Load index files and extract global IDs, byte offsets, and token lengths
        self.idx_filenames = [filename.replace('.bin', '.idx') for filename in self.shard_filenames]
        self.global_idx_list, self.byte_offset_list, self.token_length_list = [], [], []
        for idx_file in self.idx_filenames:
            with open(idx_file, 'r') as f:
                for line in f:
                    global_idx, byte_offset, token_length = line.strip().split(',')
                    self.global_idx_list.append(global_idx)
                    self.byte_offset_list.append(int(byte_offset))
                    self.token_length_list.append(int(token_length))

        # Create lookup dictionaries for byte offset and token length using global IDs
        self.byte_offset_dict = dict(zip(self.global_idx_list, self.byte_offset_list))
        self.token_length_dict = dict(zip(self.global_idx_list, self.token_length_list))

        # Load metadata
        metrics_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}" if vocab_source == "custom" else "tok0")
        print(f"Expected metadata directory: {metrics_dir}")

        self.metadata_df = pl.read_csv(os.path.join(metrics_dir, "overall_metadata_for_vocab_0.csv"))


    def __len__(self):
        return len(self.global_idx_list)

    def __getitem__(self, index):

        global_ix = index
        shard_str, row_str = global_ix.split("_")
        shard_id = int(shard_str)
        shard = self.shard_filenames[shard_id]

        byte_offset = self.byte_offset_dict[global_ix]
        token_length = self.token_length_dict[global_ix]

        m = np.memmap(shard, dtype=np.uint16, mode="r")
        start = byte_offset // np.uint16().nbytes
        end = start + token_length

        chunk = torch.from_numpy(m[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]

        # Access the metadata using polars dataframe filtering
        metadata_row = self.metadata_df.filter(self.metadata_df["global_id"] == global_ix)
        metadata = metadata_row.to_dict(as_series=False)  # Convert the filtered row to dictionary

        # Adjust the dictionary structure to match the previous format
        metadata = {col: metadata[col][0] for col in metadata}

        if "bleu_score" not in metadata:
            print(f"Missing bleu_score for global_ix: {global_ix}, metadata: {metadata}")

        return x, y, global_ix, metadata


class DynamicSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, select_func=None):
        self.dataset = dataset
        self.split = dataset.split
        self.select_func = select_func

        if self.select_func:
            self.order = self.select_func(dataset.metadata_df, dataset.global_idx_list, dataset.split)
        else:
            self.order = self.generate_order()

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)

    def generate_order(self):
        """Construct the order using the global IDs in the dataset."""
        return self.dataset.global_idx_list


def select_batches_sorted_by_column(metadata, global_idx_list, column_name, split, ascending=True):
    # Use polars to sort the metadata by the column
    metadata_sorted = metadata.sort(column_name, descending=not ascending)

    # get the global ids (last col)
    last_column_name = metadata_sorted.columns[-1]
    
    # Extracting the nested lists and flattening into a single list of indices
    sorted_2d_list = metadata_sorted.select(last_column_name).to_numpy().tolist()
    sorted_indices = [index for sublist in sorted_2d_list for index in sublist]

    # Filter based on split
    if split == "train":
        sorted_indices = [idx for idx in sorted_indices if not idx.startswith("0_")]
    else:
        sorted_indices = [idx for idx in sorted_indices if idx.startswith("0_")]
    
    # Instead of just returning the sorted global IDs, map them back to their integer index
    idx_map = {gid: i for i, gid in enumerate(global_idx_list)}
    sorted_indices_int = [idx_map[gid] for gid in sorted_indices]
    
    return sorted_indices_int


# -----------------------------------------------------------------------------
# public interface functions
def custom_collate(batch):
    # Separating inputs and labels
    x, y, global_ix, metadata = zip(*batch)
    
    # Padding the sequences
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return x, y, global_ix, metadata


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
            collate_fn=custom_collate  
        )

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True) 
            print(f"X shape: {x.shape}, Y shape: {y.shape}")
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
