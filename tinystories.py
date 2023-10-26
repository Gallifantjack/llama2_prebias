"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

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
def create_global_id(shard_id, idx):
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

def verify_bin_and_idx(bin_filename, idx_filename):
    global_ids = set()
    duplicates = []
    empties = []
    
    with open(bin_filename, "rb") as f, open(idx_filename, "r") as idx:
        for line in idx:
            global_id, byte_offset, token_length = line.strip().split(",")
            byte_offset, token_length = int(byte_offset), int(token_length)
            
            # Check for duplicate IDs
            if global_id in global_ids:
                duplicates.append(global_id)
            global_ids.add(global_id)

            # Use byte offset to jump directly to the data
            f.seek(byte_offset)
            tokens = np.frombuffer(f.read(token_length * 2), dtype=np.uint16)
            
            # Check for examples < 5 tokens
            if token_length < 5:
                empties.append(global_id)

    # Report the results
    if not duplicates:
        print("No duplicate global IDs found!")
    else:
        print(f"Found duplicate global IDs: {duplicates}")

    if not empties:
        print("No empty tokenized examples found!")
    else:
        print(f"Found empty tokenized examples for global IDs: {empties}")

def pretokenize(vocab_size):
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    os.makedirs(bin_dir, exist_ok=True)

    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        all_shard_data = list(executor.map(fun, enumerate(shard_filenames)))

    merged_tokenized_filename = os.path.join(bin_dir, "merged_data.bin")
    merged_idx_filename = os.path.join(bin_dir, "merged_data.idx")

    with open(merged_tokenized_filename, "wb") as f, open(merged_idx_filename, "w") as idx_file:
        for shard_data in all_shard_data:
            for global_id, tokens in shard_data:
                token_length = len(tokens)
                idx_file.write(f"{global_id},{f.tell()},{token_length}\n")
                f.write(np.array(tokens, dtype=np.uint16).tobytes())

    # Run the verification
    verify_bin_and_idx(merged_tokenized_filename, merged_idx_filename)

    print("Done.")

# -----------------------------------------------------------------------------
# Metadata functions
def detokenize_from_bin(bin_file, idx_file, tokenizer_path):
    enc = Tokenizer(tokenizer_model=tokenizer_path)
    texts, global_ids, empty_indices = [], [], []

    with open(bin_file, "rb") as f, open(idx_file, "r") as idx:
        for line in idx:
            global_id, byte_offset, token_length = map(int, line.strip().split(","))
            f.seek(byte_offset)
            tokens = np.frombuffer(f.read(token_length * 2), dtype=np.uint16)
            decoded_text = enc.decode(tokens.tolist())
            
            if not decoded_text.strip():
                empty_indices.append(global_id)

            texts.append(decoded_text)
            global_ids.append(global_id)

    return texts, global_ids, empty_indices

def compute_shard_metrics(bin_file, tokenizer_path, vocab_size):
    idx_file = bin_file.replace(".bin", ".idx")
    detokenized_texts, global_indices, empty_global_ids = detokenize_from_bin(bin_file, idx_file, tokenizer_path)
    
    shard_metrics = []
    for idx, detokenized_text in enumerate(detokenized_texts):
        metrics = evaluate_textual_metrics(detokenized_text, expected_stdout.decode("utf-8"))
        metrics["global_id"] = global_indices[idx]
        shard_metrics.append(metrics)

    return shard_metrics, empty_global_ids

def compute_metadata(vocab_size):
    bin_file = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}", "merged_data.bin")
    tokenizer_path = get_tokenizer_model_path(vocab_size)
    
    # Compute metrics and get empty IDs
    shard_metrics, empty_global_ids = compute_shard_metrics(bin_file, tokenizer_path, vocab_size)

    # Save results to CSV
    metrics_output_path = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}/metadata.csv")
    pl.DataFrame(shard_metrics).write_csv(metrics_output_path)
    
    empty_ids_output_path = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}/empty_ids.csv")
    pl.DataFrame({"empty_global_id": empty_global_ids}).write_csv(empty_ids_output_path)

    print(f"Processed data for vocab size: {vocab_size}")


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
            
        # Memory mapping for each shard
        self.mem_maps = [np.memmap(shard, dtype=np.uint16, mode="r") for shard in self.shard_filenames]
        
        assert self.shard_filenames, f"No bin files found in {bin_dir}"
        print(f"Number of .bin files found: {len(self.shard_filenames)}")

        # Load global index files and extract global IDs, byte offsets, and token lengths
        self.global_id_list, self.byte_offset_list, self.token_length_list = [], [], []
        with open(self.idx_filename, 'r') as f:
            for line in f:
                global_id, byte_offset, token_length = line.strip().split(',')
                # Only append if token length is >= 1
                if int(token_length) >= 1:
                    self.global_id_list.append(global_id)
                    self.byte_offset_list.append(int(byte_offset))
                    self.token_length_list.append(int(token_length))

        # Split the data into training and validation sets
        data_size = len(self.global_id_list)
        indices = list(range(data_size))
        split_idx = int(0.9 * data_size)  # 90% for training, 10% for validation
        
        # Shuffle the indices and split
        random.shuffle(indices)
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]
        
        # Select the appropriate split based on the 'split' argument
        if split == "train":
            self.global_id_list = [self.global_id_list[i] for i in train_indices]
            self.byte_offset_list = [self.byte_offset_list[i] for i in train_indices]
            self.token_length_list = [self.token_length_list[i] for i in train_indices]
        else:  # split == "valid"
            self.global_id_list = [self.global_id_list[i] for i in valid_indices]
            self.byte_offset_list = [self.byte_offset_list[i] for i in valid_indices]
            self.token_length_list = [self.token_length_list[i] for i in valid_indices]


        # Create lookup dictionaries for byte offset and token length using global IDs
        self.byte_offset_dict = dict(zip(self.global_id_list, self.byte_offset_list))
        self.token_length_dict = dict(zip(self.global_id_list, self.token_length_list))

        # Load metadata
        metrics_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}" if vocab_source == "custom" else "tok0")
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


    def __len__(self):
        return len(self.global_id_list)

    def __getitem__(self, global_id_value):
        global_ix = global_id_value
        shard_str, _ = global_ix.split("_")
        shard_id = int(shard_str)
        
        byte_offset = self.byte_offset_dict[global_ix]
        token_length = self.token_length_dict[global_ix]

        m = self.mem_maps[shard_id]
        start = byte_offset // np.uint16().nbytes
        end = start + token_length

        chunk = torch.from_numpy(m[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]

        # Access the metadata using the pre-built dictionary index
        metadata = self.metadata_dict.get(global_ix, {})
                                
        return x, y, global_ix, metadata


class DynamicSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, split, select_func=None):
        self.dataset = dataset
        self.split = split
        self.select_func = select_func
        self.global_id_list = dataset.global_id_list

        if self.select_func:
            self.order = self.select_func(self.dataset.metadata_df, self.global_id_list, self.split)
        else:
            self.order = self.generate_order()


    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)

    def generate_order(self):
        """Construct the order using the global IDs in the dataset if no select_func is provided"""
        return self.global_id_list


def select_batches_sorted_by_column(metadata, global_id_list, column_name, split, ascending=True):
    # Use polars to sort the metadata by the column
    metadata_sorted = metadata.sort(column_name, descending=not ascending)

    # get the global ids (last col)
    last_column_name = metadata_sorted.columns[-1]
    
    # Extracting the nested lists and flattening into a single list of indices
    sorted_2d_list = metadata_sorted.select(last_column_name).to_numpy().tolist()
    sorted_indices = [global_index_value for sublist in sorted_2d_list for global_index_value in sublist]

    # Filter based on split
    if split == "train":
        sorted_indices = [idx for idx in sorted_indices if not idx.startswith("0_")]
    else:
        sorted_indices = [idx for idx in sorted_indices if idx.startswith("0_")]
    
    # Instead of just returning the sorted global IDs, map them back to their integer index
    idx_map = {gid: i for i, gid in enumerate(global_id_list)}
    sorted_indices_int = [idx_map[gid] for gid in sorted_indices]
    
    print(f"Sorted indices: {sorted_indices_int}")
    
    return sorted_indices_int


# -----------------------------------------------------------------------------
# public interface functions
def custom_collate(batch, MAX_SEQ_LEN):
    # Separating inputs and labels
    x, y, global_ix, metadata = zip(*batch)
    
    # Limit sequence length to max_seq_len
    x = [seq[:MAX_SEQ_LEN] for seq in x]
    y = [seq[:MAX_SEQ_LEN] for seq in y]
    
    # Padding the sequences
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return x, y, global_ix, metadata

def get_custom_collate(max_seq_len):
    return lambda batch: custom_collate(batch, max_seq_len)

class Task:
    @staticmethod
    def iter_batches(
        split, batch_size, device, num_workers=0, select_func=None, **dataset_kwargs
    ):
        ds = PretokDataset(split=split, **dataset_kwargs)
        
        # If select_func is provided, don't shuffle. Otherwise, shuffle the data.
        sampler = DynamicSampler(ds, split= split, select_func=select_func)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=get_custom_collate(max_seq_len=128)  # TODO: make this configurable
        )

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True) 
            
            print(f"global_ix: {global_ix}, metadata: {metadata}")
            print(f"x: {x}, y: {y}")
                        
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
