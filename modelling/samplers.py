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

from utils.paths import DATA_CACHE_DIR
from utils.functions import create_global_id, get_tokenizer_model_path


class DynamicSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, split, select_func=None):
        self.dataset = dataset
        self.split = split
        self.select_func = select_func
        self.global_id_list = dataset.global_id_list

        if self.select_func:
            self.order = self.select_func(
                self.dataset.metadata_df, self.global_id_list, self.split
            )
        else:
            self.order = self.generate_order()

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)

    def generate_order(self):
        """Construct the order using the global IDs in the dataset if no select_func is provided"""
        return self.global_id_list


def select_batches_sorted_by_column(
    metadata, global_id_list, column_name, split, ascending=True
):
    # Use polars to sort the metadata by the column
    metadata_sorted = metadata.sort(column_name, descending=not ascending)

    # get the global ids (last col)
    last_column_name = metadata_sorted.columns[-1]

    # Extracting the nested lists and flattening into a single list of indices
    sorted_2d_list = metadata_sorted.select(last_column_name).to_numpy().tolist()
    sorted_indices = [
        global_index_value
        for sublist in sorted_2d_list
        for global_index_value in sublist
    ]

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
