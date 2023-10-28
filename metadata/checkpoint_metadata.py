import os
import torch
from modelling.model import Transformer, ModelArgs
from train_tok.tokenizer import Tokenizer
import polars as pl
from metadata.evaluators import evaluate_textual_metrics
from itertools import chain
from pathlib import Path

# Import necessary utilities
from utils.paths import DATA_CACHE_DIR
from metadata.batch_metadata import expected_stdout
from utils.functions import get_tokenizer_model_path

# -----------------------------------------------------------------------------
# test utilities


def load_model_from_checkpoint(checkpoint_file):
    checkpoint_dict = torch.load(checkpoint_file, map_location="cpu")
    # load batch info
    batch_info = checkpoint_dict["batch_indices_trained"]

    model_args = ModelArgs(
        **checkpoint_dict["model_args"]
    )  # Adjusted this to match the test script
    model = Transformer(model_args)

    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    # load model state dict
    model.load_state_dict(state_dict, strict=False)

    return model, batch_info


def extract_checkpoint_number(filename):
    try:
        # Updated extraction based on the 'ckpt_NUMBER.pt' format
        return int(filename.split("ckpt_")[1].split(".pt")[0])
    except IndexError:
        print(f"Unexpected filename structure: {filename}")
        return None


# -----------------------------------------------------------------------------
# test function


def evaluate_model(model, tokenizer_path):
    x = torch.tensor([[1]], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=200, temperature=0.0)
    pt_tokens = y[0].tolist()
    enc = Tokenizer(tokenizer_model=tokenizer_path)
    generated_text = enc.decode(pt_tokens)

    metrics = evaluate_textual_metrics(generated_text, expected_stdout.decode("utf-8"))
    metrics["text"] = generated_text
    return metrics


def run_evaluation(out_dir, vocab_size):
    tokenizer_filepath = get_tokenizer_model_path(vocab_size=vocab_size)
    parquet_file_path = os.path.join(
        DATA_CACHE_DIR, f"tok{vocab_size}", "merged_data_with_metadata.parquet"
    )
    checkpoint_directory = os.path.join(out_dir, "ckpt")
    checkpoint_files = list(Path(checkpoint_directory).glob("*.pt"))

    checkpoint_output_results = []
    all_batch_results = []

    # Step 1: Adjust the Reading Mechanism
    df_metrics = pl.read_parquet(parquet_file_path)

    for idx, checkpoint_file in enumerate(checkpoint_files):
        print(f"Evaluating checkpoint {idx+1}/{len(checkpoint_files)}")
        model, batch_info = load_model_from_checkpoint(checkpoint_file)
        result = evaluate_model(model, tokenizer_filepath)
        checkpoint_name = os.path.basename(checkpoint_file)
        checkpoint_name = extract_checkpoint_number(checkpoint_name)
        result["checkpoint_name"] = checkpoint_name
        checkpoint_output_results.append(result)

        checkpoint_df = pl.DataFrame({"checkpoint_name": [checkpoint_name]})
        batch_info = list(chain(*[tensor.tolist() for tensor in batch_info]))

        # Step 2: Filter to only batches used at a checkpoint
        df_filtered = df_metrics.filter(df_metrics["id"].is_in(batch_info))

        # Step 3: Select only the metrics columns
        excluded_columns = ["id", "tokens"]
        metrics = [col for col in df_filtered.columns if col not in excluded_columns]
        df_filtered = df_filtered.select(metrics)

        # Step 4: Calculate the mean of the metrics at a checkpoint
        batch_results = df_filtered.mean().hstack(checkpoint_df)

        all_batch_results.append(batch_results)  # Append batch results

    # Combine all batch results
    batch_results_df = pl.concat(all_batch_results)

    # Save to Parquet
    metadata_dir = os.path.join(out_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    checkpoint_output_results_df = pl.DataFrame(checkpoint_output_results)
    checkpoint_output_results_df.write_parquet(
        os.path.join(metadata_dir, "checkpoint_output_results.parquet")
    )
    batch_results_df.write_parquet(os.path.join(metadata_dir, "batch_results.parquet"))

    print("Evaluation Done!")
