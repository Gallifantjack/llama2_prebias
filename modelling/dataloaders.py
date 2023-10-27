from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
import polars as pl
import pandas as pd
from torch.utils.data import Sampler

# relative imports
from modelling.samplers import DynamicSampler
from modelling.pretokdataset import PretokDataset


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
        sampler = DynamicSampler(ds, split=split, select_func=select_func)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=get_custom_collate(
                max_seq_len=128
            ),  # TODO: make this configurable
        )

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            print(f"global_ix: {global_ix}, metadata: {metadata}")
            print(f"x: {x}, y: {y}")

            yield x, y, global_ix, metadata
