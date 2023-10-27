import pandas as pd
import pytest
import polars as pl
import numpy as np
import glob
import os


def test_no_duplicate_global_ids(data_cache_dir, vocab_size_abs):
    parquet_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}", "merged_data.parquet"
    )

    # Load the Parquet file
    df = pd.read_parquet(parquet_path)

    # Check for duplicate IDs
    duplicates = df[df.duplicated(subset=["id"], keep=False)]
    assert (
        duplicates.shape[0] == 0
    ), f"Found duplicate global IDs: {duplicates['id'].tolist()}"


def test_no_empty_tokenized_examples(data_cache_dir, vocab_size_abs):
    parquet_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}", "merged_data.parquet"
    )

    # Load the Parquet file
    df = pd.read_parquet(parquet_path)

    # Check for examples < 5 tokens
    empties = df[df["tokens"].apply(lambda x: len(x) < 5)]
    assert (
        empties.shape[0] == 0
    ), f"Found empty tokenized examples for global IDs: {empties['id'].tolist()}"


def test_all_entries_have_both_tokens_and_id(data_cache_dir, vocab_size_abs):
    parquet_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}", "merged_data.parquet"
    )

    # Load the Parquet file
    df = pd.read_parquet(parquet_path)

    # Check that all entries have tokens and an id
    missing_ids = df[df["id"].isna()]
    missing_tokens = df[df["tokens"].isna()]

    assert missing_ids.shape[0] == 0, f"Entries without IDs: {missing_ids.shape[0]}"
    assert (
        missing_tokens.shape[0] == 0
    ), f"Entries without tokens: {missing_tokens.shape[0]}"
