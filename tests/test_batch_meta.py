import pandas as pd
import os
import pytest


def load_dataframe(data_cache_dir, vocab_size_abs):
    parquet_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}", "merged_data_with_metadata.parquet"
    )
    return pd.read_parquet(parquet_path)


def test_no_duplicate_ids(data_cache_dir, vocab_size_abs):
    df = load_dataframe(data_cache_dir, vocab_size_abs)
    duplicate_ids = df[df.duplicated(subset="id")]
    assert duplicate_ids.empty, f"Found {duplicate_ids.shape[0]} duplicate IDs!"


def test_metadata_completeness(data_cache_dir, vocab_size_abs):
    df = load_dataframe(data_cache_dir, vocab_size_abs)

    # Given metrics columns
    metadata_columns = [
        "bleu_score",
        "flesch_kincaid_grade",
        "gunning_fog",
        "vocabulary_diversity",
        "subjectivity_score",
        "sentiment_score",
        "profanity_check",
    ]

    rows_with_missing_values = df[df[metadata_columns].isnull().any(axis=1)]

    assert rows_with_missing_values.empty, (
        f"Found {rows_with_missing_values.shape[0]} rows with missing values! Here are some of them: \n"
        f"{rows_with_missing_values.head(10)}"
    )


def test_no_nan_values(data_cache_dir, vocab_size_abs):
    df = load_dataframe(data_cache_dir, vocab_size_abs)
    null_values_count = df.isnull().sum().sum()  # Sum over all columns and rows
    assert (
        null_values_count == 0
    ), f"There are {null_values_count} NaN/missing values in the dataframe!"


def load_original_dataframe(data_cache_dir, vocab_size_abs):
    parquet_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}", "merged_data.parquet"
    )
    return pd.read_parquet(parquet_path)


def test_matching_ids(data_cache_dir, vocab_size_abs):
    df_metadata = load_dataframe(data_cache_dir, vocab_size_abs)
    df_original = load_original_dataframe(data_cache_dir, vocab_size_abs)

    # Check that the number of rows match
    assert (
        df_metadata.shape[0] == df_original.shape[0]
    ), f"Number of rows mismatch! Metadata: {df_metadata.shape[0]}, Original: {df_original.shape[0]}."

    # Check if IDs match in both dataframes
    assert (
        df_metadata["id"] == df_original["id"]
    ).all(), "IDs in the metadata and original dataframes do not match!"
