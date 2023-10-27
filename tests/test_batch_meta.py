import numpy as np
import pytest
import os
import polars as pl

from metadata.batch_metadata import detokenize_from_bin


# check concordance between metadata and idx file
@pytest.fixture
def computed_metrics_and_indices(data_cache_dir, vocab_size_abs):
    metrics_output_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}/metadata.csv"
    )
    empty_ids_output_path = os.path.join(
        data_cache_dir, f"tok{vocab_size_abs}/empty_ids.csv"
    )
    idx_file_path = os.path.join(data_cache_dir, f"tok{vocab_size_abs}/merged_data.idx")

    # Read global ids from metadata
    shard_metrics = pl.read_csv(metrics_output_path).to_pandas().to_dict("records")
    global_ids_from_metadata = [metric["global_id"] for metric in shard_metrics]

    # Read empty global ids
    empty_global_ids = (
        pl.read_csv(empty_ids_output_path).to_pandas()["empty_global_id"].tolist()
    )

    # Read global ids from idx file
    with open(idx_file_path, "r") as idx_file:
        global_ids_from_idx = [
            int(line.strip().split(",")[0]) for line in idx_file.readlines()
        ]

    return global_ids_from_metadata, empty_global_ids, global_ids_from_idx


def test_empty_ids_not_in_metadata(computed_metrics_and_indices):
    global_ids_from_metadata, empty_global_ids, _ = computed_metrics_and_indices
    assert not any(
        id in global_ids_from_metadata for id in empty_global_ids
    ), "Some empty global ids were found in metadata."


def test_metadata_plus_empty_equals_idx(computed_metrics_and_indices):
    (
        global_ids_from_metadata,
        empty_global_ids,
        global_ids_from_idx,
    ) = computed_metrics_and_indices
    assert len(set(global_ids_from_metadata).union(set(empty_global_ids))) == len(
        set(global_ids_from_idx)
    ), "The total unique ids in metadata and empty ids do not match the total in the idx file."


def test_no_duplicates_in_metadata(computed_metrics_and_indices):
    global_ids_from_metadata, _, _ = computed_metrics_and_indices
    assert len(global_ids_from_metadata) == len(
        set(global_ids_from_metadata)
    ), "Duplicate global ids found in metadata.csv"


def test_no_duplicates_in_idx(computed_metrics_and_indices):
    _, _, global_ids_from_idx = computed_metrics_and_indices
    assert len(global_ids_from_idx) == len(
        set(global_ids_from_idx)
    ), "Duplicate global ids found in idx file."


# empty ids


# emptt metadata

# test tokenizer


# def test_detokenize_from_bin(mock_data):
#     bin_file, idx_file, tokenizer_path = mock_data

#     expected_texts = ["hello"]
#     expected_global_ids = [0]
#     expected_empty_indices = []

#     texts, global_ids, empty_indices = detokenize_from_bin(
#         bin_file, idx_file, tokenizer_path
#     )

#     assert texts == expected_texts, f"Expected {expected_texts}, but got {texts}"
#     assert (
#         global_ids == expected_global_ids
#     ), f"Expected {expected_global_ids}, but got {global_ids}"
#     assert (
#         empty_indices == expected_empty_indices
#     ), f"Expected {expected_empty_indices}, but got {empty_indices}"


# Run pytest
if __name__ == "__main__":
    pytest.main()
