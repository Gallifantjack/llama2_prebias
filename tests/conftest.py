import pytest
import sys
import os
import polars as pl

from utils.paths import DATA_CACHE_DIR
from utils.functions import get_tokenizer_model_path


@pytest.fixture
def data_cache_dir():
    return DATA_CACHE_DIR


@pytest.fixture
def vocab_size_abs():
    return 0


# ----------------------------
# batch metadata test fixtures

# Mock data
MOCK_BIN_DATA = b"\x01\x00\x02\x00\x03\x00"  # Represents tokens [1, 2, 3]
MOCK_IDX_DATA = "0,0,3\n"


# Setup function to prepare mock data
@pytest.fixture(scope="module")
def mock_data():
    # Create mock bin and idx files
    with open("mock_data.bin", "wb") as f:
        f.write(MOCK_BIN_DATA)
    with open("mock_data.idx", "w") as f:
        f.write(MOCK_IDX_DATA)
    tokenizer_path = get_tokenizer_model_path(
        vocab_size=0
    )  # Assuming this is how you fetch the tokenizer
    yield tokenizer_path  # This is the value that will be passed to the test function
    # Cleanup after tests
    os.remove("mock_data.bin")
    os.remove("mock_data.idx")
