import pytest
import torch
import os
import pandas as pd


from modelling.pretokdataset import PretokDataset
from modelling.samplers import DynamicSampler
from modelling.dataloaders import Task
from modelling.transformation_functions import *


# Helper function to create a sample parquet file for testing
import random


def create_sample_data(sample_parquet_path):
    df = pd.DataFrame(
        {
            "id": list(range(10)),
            "tokens": [[random.randint(0, 100) for _ in range(3)] for _ in range(10)],
            "bleu_score": [random.random() for _ in range(10)],
            "flesch_kincaid_grade": [random.randint(0, 10) for _ in range(10)],
            "gunning_fog": [random.randint(0, 10) for _ in range(10)],
            "vocabulary_diversity": [random.random() for _ in range(10)],
            "subjectivity_score": [random.random() for _ in range(10)],
            "sentiment_score": [random.random() for _ in range(10)],
            "profanity_check": [random.random() for _ in range(10)],
        }
    )
    df.to_parquet(sample_parquet_path)


# Test PretokDataset
def test_PretokDataset(sample_parquet_path):
    create_sample_data(sample_parquet_path)

    dataset = PretokDataset(
        split="train", max_seq_len=3, vocab_size=0, vocab_source="test"
    )
    assert len(dataset) == 9

    dataset_valid = PretokDataset(
        split="valid", max_seq_len=3, vocab_size=0, vocab_source="test"
    )
    assert len(dataset_valid) == 1


# Test DynamicSampler
def test_DynamicSampler_transform(sample_parquet_path):
    create_sample_data(sample_parquet_path)
    dataset = PretokDataset(
        split="train", max_seq_len=3, vocab_size=0, vocab_source="test"
    )
    reverse_order_fn = lambda df: df.index.tolist()[::-1]

    sampler = DynamicSampler(dataset, split="train", transform_func=reverse_order_fn)
    order = list(sampler)
    assert order == list(range(8, -1, -1))


def test_sort_ascending(sample_parquet_path):
    create_sample_data(sample_parquet_path)
    df = pd.read_parquet(sample_parquet_path)

    sorted_df = sort_ascending(df, "bleu_score")
    assert (sorted_df["bleu_score"] == sorted(df["bleu_score"])).all()


def test_sort_descending(sample_parquet_path):
    create_sample_data(sample_parquet_path)
    df = pd.read_parquet(sample_parquet_path)

    sorted_df = sort_descending(df, "bleu_score")
    assert (sorted_df["bleu_score"] == sorted(df["bleu_score"], reverse=True)).all()


def test_filter_by_threshold(sample_parquet_path):
    create_sample_data(sample_parquet_path)
    df = pd.read_parquet(sample_parquet_path)

    threshold = 5
    filtered_df = filter_by_threshold(df, "bleu_score", threshold)
    assert (filtered_df["bleu_score"] > threshold).all()


def test_combined_transform(sample_parquet_path):
    create_sample_data(sample_parquet_path)
    df = pd.read_parquet(sample_parquet_path)

    threshold = 5
    transformed_df = combined_transform(df, "bleu_score", threshold)
    assert (transformed_df["bleu_score"] > threshold).all()
    assert (transformed_df["bleu_score"] == sorted(transformed_df["bleu_score"])).all()


# Test Task.iter_batches
def test_Task_iter_batches(sample_parquet_path):
    create_sample_data(sample_parquet_path)
    for x, y, global_ix, metadata in Task.iter_batches(
        split="train",
        batch_size=2,
        device=torch.device("cpu"),
        max_seq_len=3,
        vocab_size=0,
        vocab_source="test",
    ):
        assert "bleu_score" in metadata


# If the file is executed as a script, run the tests
if __name__ == "__main__":
    pytest.main()
