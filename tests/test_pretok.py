import pandas as pd
import pytest
import polars as pl
import numpy as np
import glob
import os


def test_no_duplicate_global_ids(data_cache_dir, vocab_size_abs):
    bin_dir = os.path.join(data_cache_dir, f"tok{vocab_size_abs}")
    merged_idx_filename = os.path.join(bin_dir, "merged_data.idx")

    global_ids = set()
    duplicates = []

    with open(merged_idx_filename, "r") as idx:
        for line in idx:
            global_id, _, _ = line.strip().split(",")
            if global_id in global_ids:
                duplicates.append(global_id)
            global_ids.add(global_id)

    assert len(duplicates) == 0, f"Found duplicate global IDs: {duplicates}"


def test_no_empty_tokenized_examples(data_cache_dir, vocab_size_abs):
    bin_dir = os.path.join(data_cache_dir, f"tok{vocab_size_abs}")
    merged_tokenized_filename = os.path.join(bin_dir, "merged_data.bin")
    merged_idx_filename = os.path.join(bin_dir, "merged_data.idx")

    empties = []

    with open(merged_tokenized_filename, "rb") as f, open(
        merged_idx_filename, "r"
    ) as idx:
        for line in idx:
            global_id, byte_offset, token_length = line.strip().split(",")
            byte_offset, token_length = int(byte_offset), int(token_length)

            # Use byte offset to jump directly to the data
            f.seek(byte_offset)
            tokens = np.frombuffer(f.read(token_length * 2), dtype=np.uint16)

            # Check for examples < 5 tokens
            if token_length < 5:
                empties.append(global_id)

    assert (
        len(empties) == 0
    ), f"Found empty tokenized examples for global IDs: {empties}"
