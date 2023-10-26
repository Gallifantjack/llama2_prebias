import polars as pl
import glob
import os
import pstats

from tinystories import PretokDataset

# # profiler
# output = "output_profile_stats"
# p = pstats.Stats(output)
# p.sort_stats('cumulative').print_stats(20)  # This will print the top 20 functions sorted by cumulative time


def check_id_discrepancies(idx_filenames, metadata_df):
    # Extract all global IDs from the .idx files
    idx_global_ids = []
    for idx_file in idx_filenames:
        with open(idx_file, 'r') as f:
            for line in f:
                global_id, _, _ = line.strip().split(',')
                idx_global_ids.append(global_id)

    # Extract all global IDs from the metadata dataframe
    metadata_global_ids = set(metadata_df["global_id"].to_list())

    # Find global IDs that are in the .idx files but not in the metadata
    missing_in_metadata = [gid for gid in idx_global_ids if gid not in metadata_global_ids]

    # Find global IDs that are in the metadata but not in the .idx files
    missing_in_idx = [gid for gid in metadata_global_ids if gid not in idx_global_ids]

    return missing_in_metadata, missing_in_idx

# Use the check_id_discrepancies function
ds = PretokDataset(split="train", max_seq_len=128, vocab_size=0, vocab_source="custom") # Example arguments, adjust accordingly
missing_in_metadata, missing_in_idx = check_id_discrepancies(ds.idx_filenames, ds.metadata_df)

if not missing_in_metadata and not missing_in_idx:
    print("The global IDs in the .idx files and metadata match perfectly!")
else:
    if missing_in_metadata:
        print(f"Global IDs present in .idx files but missing in metadata: {missing_in_metadata}")
    if missing_in_idx:
        print(f"Global IDs present in metadata but missing in .idx files: {missing_in_idx}")
