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

from train_tok.tokenizer import Tokenizer
import polars as pl
import pandas as pd
from torch.utils.data import Sampler

# relative imports
from utils.paths import DATA_CACHE_DIR
from utils.functions import create_global_id, get_tokenizer_model_path


class PretokDataset(torch.utils.data.Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(
        self,
        split,
        max_seq_len,
        vocab_size,
        vocab_source,
        **kwargs,
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

        # Load parquet data
        parquet_dir = os.path.join(
            DATA_CACHE_DIR,
            f"tok{self.vocab_size}" if vocab_source == "custom" else "tok0",
            "merged_data_with_metadata.parquet",
        )
        self.data_df = pd.read_parquet(parquet_dir)

        # Set the 'id' column as the index for efficient row access
        self.data_df.set_index("id", inplace=True)

        # Split the data into training and validation sets
        data_size = len(self.data_df)
        split_idx = int(0.9 * data_size)  # 90% for training, 10% for validation

        if split == "train":
            self.data_df = self.data_df.iloc[:split_idx]
        else:  # split == "valid"
            self.data_df = self.data_df.iloc[split_idx:]

        # Convert the entire tokens column to a list of tensors once during initialization
        self.data_df["tokens"] = self.data_df["tokens"].apply(
            lambda x: torch.tensor(x, dtype=torch.int64)
        )

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # Here, idx will be a global_id
        row = self.data_df.loc[idx]
        global_ix = row.name  # Since 'id' is now the index, we use .name to get it

        tokens = row["tokens"]
        x = tokens[:-1]
        y = tokens[1:]

        # Extract metadata directly from the row
        metadata = {
            col: row[col]
            for col in [
                "bleu_score",
                "flesch_kincaid_grade",
                "gunning_fog",
                "vocabulary_diversity",
                "subjectivity_score",
                "sentiment_score",
                "profanity_check",
            ]
        }

        return x, y, global_ix, metadata
