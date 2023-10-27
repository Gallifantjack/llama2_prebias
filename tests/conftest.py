import pytest
import sys

from utils.paths import DATA_CACHE_DIR


@pytest.fixture
def data_cache_dir():
    return DATA_CACHE_DIR


@pytest.fixture
def vocab_size_abs():
    return 0
