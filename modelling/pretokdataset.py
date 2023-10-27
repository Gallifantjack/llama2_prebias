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

# relative imports
from utils.paths import DATA_CACHE_DIR
from utils.functions import create_global_id, get_tokenizer_model_path


class PretokDataset(torch.utils.data.Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, select_func=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

        # Determine the appropriate directory for .bin files
        bin_dir = os.path.join(
            DATA_CACHE_DIR,
            f"tok{self.vocab_size}" if vocab_source == "custom" else "tok0",
        )
        print(f"Expected .bin file directory: {bin_dir}")

        self.shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        if split == "train":
            self.shard_filenames = self.shard_filenames[1:]
        else:
            self.shard_filenames = self.shard_filenames[:1]

        # Memory mapping for each shard
        self.mem_maps = [
            np.memmap(shard, dtype=np.uint16, mode="r")
            for shard in self.shard_filenames
        ]

        assert self.shard_filenames, f"No bin files found in {bin_dir}"
        print(f"Number of .bin files found: {len(self.shard_filenames)}")

        # Load global index files and extract global IDs, byte offsets, and token lengths
        self.global_id_list, self.byte_offset_list, self.token_length_list = [], [], []
        with open(self.idx_filename, "r") as f:
            for line in f:
                global_id, byte_offset, token_length = line.strip().split(",")
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
        metrics_dir = os.path.join(
            DATA_CACHE_DIR,
            f"tok{self.vocab_size}" if vocab_source == "custom" else "tok0",
        )
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
