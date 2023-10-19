import os
import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import polars as pl
from evaluators import evaluate_textual_metrics
from itertools import chain


# Global constants
checkpoint_dir = "out/ckpt/"
tokenizer_path = "tokenizer.model"
expected_stdout = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"
metrics_csv_path = "data/TinyStories_all_data/batch_metrics.csv"

# -----------------------------------------------------------------------------
# test utilities


def load_model_from_checkpoint(checkpoint_file):
    checkpoint_dict = torch.load(checkpoint_file, map_location="cpu")
    # load batch info
    batch_info = checkpoint_dict["batch_indices_trained"]
    print(f'Batch indices trained on for checkpoint "{checkpoint_file}":')
    print(batch_info)

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


def run_evaluation(checkpoint_directory, tokenizer_filepath, metrics_csv_filepath):
    checkpoint_files = [
        os.path.join(checkpoint_directory, file)
        for file in os.listdir(checkpoint_directory)
        if file.endswith(".pt")
    ]

    checkpoint_output_results = []
    checkpoint_batch_results = pl.DataFrame()

    df_metrics = pl.read_csv(metrics_csv_filepath)
    print(df_metrics.tail(5))

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
        df_filtered = df_metrics.filter(df_metrics["global_idx"].is_in(batch_info))
        batch_results = df_filtered.mean()
        batch_results = batch_results.hstack(checkpoint_df)
        checkpoint_batch_results = checkpoint_batch_results.vstack(batch_results)

    # Determine directory name for the checkpoint for output structure
    checkpoint_dir_name = os.path.basename(os.path.normpath(checkpoint_directory))

    # Save the summary results based on the checkpoint directory
    summary_output_path = f"out/tables/{checkpoint_dir_name}/summary.csv"
    if not os.path.exists(os.path.dirname(summary_output_path)):
        os.makedirs(os.path.dirname(summary_output_path))
    output_df = pl.DataFrame(checkpoint_output_results)
    output_df.write_csv(summary_output_path)

    # Save the batch results based on the checkpoint directory
    batch_results_output_path = f"out/tables/{checkpoint_dir_name}/batch_results.csv"
    if not os.path.exists(os.path.dirname(batch_results_output_path)):
        os.makedirs(os.path.dirname(batch_results_output_path))
    print(checkpoint_batch_results)
    checkpoint_batch_results.write_csv(batch_results_output_path)

    print("Evaluation Done!")


# -----------------------------------------------------------------------------
# run evaluation

if __name__ == "__main__":
    run_evaluation(checkpoint_dir, tokenizer_path, metrics_csv_path)
