import polars as pl
import glob
import os
import pstats

from tinystories import PretokDataset

# # profiler
# output = "output_profile_stats"
# p = pstats.Stats(output)
# p.sort_stats('cumulative').print_stats(20)  # This will print the top 20 functions sorted by cumulative time

import numpy as np

def load_example(bin_file, byte_offset, token_length):
    """Load an example from the binary file."""
    bin_file.seek(byte_offset)
    tokens = np.frombuffer(bin_file.read(token_length * np.uint16().nbytes), dtype=np.uint16)
    return tokens

def debug_empty_example(shard_id, bin_filename, idx_filename):
    """Check each example in the given shard's bin and idx files for emptiness."""
    
    with open(bin_filename, "rb") as bin_file, open(idx_filename, "r") as idx_file:
        for line in idx_file:
            global_id, byte_offset, token_length = line.strip().split(',')
            byte_offset, token_length = int(byte_offset), int(token_length)
            
            tokens = load_example(bin_file, byte_offset, token_length)
            
            if len(tokens) == 0:
                print(f"Found empty example with global_id: {global_id}")
                return True
            
    print("No empty examples found.")
    return False

# Update the paths to your files accordingly
bin_dir = "data/tok0" # Replace with your appropriate paths
shard_basename = "data02.bin"  # Name of the second shard without file extension

bin_filename = os.path.join(bin_dir, shard_basename.replace(".json", ".bin"))
idx_filename = os.path.join(bin_dir, shard_basename.replace(".json", ".idx"))

debug_empty_example(1, bin_filename, idx_filename) # Using shard_id=1 for the second shard
