import cProfile

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

import pandas as pd
from torch.utils.data import Sampler

from utils.paths import DATA_CACHE_DIR
from utils.functions import create_global_id, get_tokenizer_model_path


class DynamicSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, split, transform_func=None):
        self.dataset = dataset
        self.split = split
        self.transform_func = transform_func

        # If transform_func is given, utilize it to generate the order
        # Otherwise, directly use the index of the dataframe
        self.order = (
            self.transform_func(self.dataset.data_df)
            if self.transform_func
            else self.dataset.data_df.index.tolist()
        )

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)
