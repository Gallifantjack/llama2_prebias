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
        )

        for x, y, global_ix, metadata in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            print(f"global_ix: {global_ix}, metadata: {metadata}")
            print(f"x: {x}, y: {y}")

            yield x, y, global_ix, metadata
