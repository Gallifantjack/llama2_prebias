import os

from utils.paths import DATA_CACHE_DIR


def create_global_id(shard_id, idx):
    # A simple method to create unique ID by combining shard_id and idx
    # You can modify this function if needed.
    return f"{shard_id}_{idx}"


def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")
