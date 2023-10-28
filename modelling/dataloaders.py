from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from train_tok.tokenizer import Tokenizer
import polars as pl
import pandas as pd
from torch.utils.data import Sampler

# relative imports
from modelling.samplers import DynamicSampler
from modelling.pretokdataset import PretokDataset


def collate_fn(batch):
    x, y, global_ix, metadata = zip(*batch)

    # Stack the data. Assumes x and y are tensors.
    x = torch.stack(x)
    y = torch.stack(y)

    return x, y, global_ix, metadata


class Task:
    @staticmethod
    def iter_batches(
        split, batch_size, device, num_workers=0, transform_func=None, **dataset_kwargs
    ):
        ds = PretokDataset(split=split, **dataset_kwargs)

        # If select_func is provided, don't shuffle. Otherwise, shuffle the data.
        sampler = DynamicSampler(ds, split=split, transform_func=transform_func)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            yield x, y, global_ix, metadata
