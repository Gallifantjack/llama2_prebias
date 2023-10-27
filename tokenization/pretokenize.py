from utils.paths import DATA_CACHE_DIR
from utils.functions import get_tokenizer_model_path
import numpy as np

import json
import glob
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from train_tok.tokenizer import Tokenizer


class PreprocessingPipeline:
    def __init__(
        self, tokenizer_model_path, data_cache_dir, vocab_size, max_seq_length
    ):
        self.tokenizer_model_path = tokenizer_model_path
        self.data_cache_dir = data_cache_dir
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer(self.tokenizer_model_path)
        self.global_id_counter = 0

    def extract_stories(self, shard):
        with open(shard, "r") as f:
            data = json.load(f)
        return [example["story"] for example in data]

    def tokenize_story(self, story):
        return self.tokenizer.encode(story.strip(), bos=True, eos=False)

    def chunk_data(self, data, chunk_size):
        """Divide data into chunks."""
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    def parallel_tokenize(self, stories_chunk):
        """Tokenize a chunk of stories."""
        return [self.tokenize_story(story) for story in stories_chunk]

    def batch_and_pad(self, tokens_list):
        batches = []
        for tokens in tokens_list:
            if len(tokens) > self.max_seq_length:
                tokens = tokens[: self.max_seq_length]
            else:
                padding_length = self.max_seq_length - len(tokens)
                tokens.extend([0] * padding_length)  # Assuming 0 is the padding token
            batches.append(np.array(tokens, dtype=np.int32))
        return batches

    def process_shard(self, shard):
        stories = self.extract_stories(shard)

        # Divide stories into chunks
        chunk_size = len(stories) // 30  # assuming 30 cores
        story_chunks = list(self.chunk_data(stories, chunk_size))

        # Parallelize tokenization across chunks
        with ProcessPoolExecutor() as executor:
            tokenized_stories_chunks = list(
                executor.map(self.parallel_tokenize, story_chunks)
            )

        # Flatten the tokenized stories
        tokenized_stories = [
            story for chunk in tokenized_stories_chunks for story in chunk
        ]

        batches = self.batch_and_pad(tokenized_stories)
        return batches

    def run(self, debug=False):
        data_dir = os.path.join(self.data_cache_dir, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

        if debug:
            shard_filenames = shard_filenames[:2]

        all_batches = []
        for shard in tqdm(shard_filenames):
            batches = self.process_shard(shard)
            all_batches.extend(batches)

        # Create a DataFrame and write to Parquet
        start_id = self.global_id_counter
        end_id = start_id + len(all_batches)
        df = pd.DataFrame(
            {
                "id": range(start_id, end_id),
                "tokens": all_batches,
            }
        )

        parquet_path = os.path.join(
            self.data_cache_dir, f"tok{self.vocab_size}", "merged_data.parquet"
        )
        df.to_parquet(parquet_path, compression="snappy", index=False)
        self.global_id_counter = end_id
        print("Done.")


if __name__ == "__main__":
    # Set up paths
    data_cache_dir = DATA_CACHE_DIR
    vocab_size = 0
    tokenizer_model_path = get_tokenizer_model_path(vocab_size)
    max_seq_length = 128

    # Run the pipeline
    pipeline = PreprocessingPipeline(
        tokenizer_model_path, data_cache_dir, vocab_size, max_seq_length
    )
    pipeline.run(debug=True)


# what are we parallelising in the above script, as we have multiple processes running, but very little usage and tqdm stuck on 0% .

# check for errors in the above script. we are looking for unique ids and efficient storage of token batches that can be indexed by these unique ids
