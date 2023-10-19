import os
import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import polars as pl
from evaluators import evaluate_textual_metrics
from itertools import chain

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


def evaluate_all_checkpoints(checkpoint_dir, tokenizer_path, metrics_csv_path):
    checkpoint_files = [
        os.path.join(checkpoint_dir, file)
        for file in os.listdir(checkpoint_dir)
        if file.endswith(".pt")
    ]

    checkpoint_output_results = []
    checkpoint_batch_results = pl.DataFrame()

    df_metrics = pl.read_csv(metrics_csv_path)
    # print first 5 rows
    print(df_metrics.tail(5))

    for idx, checkpoint_file in enumerate(checkpoint_files):
        print(f"Evaluating checkpoint {idx+1}/{len(checkpoint_files)}")
        model, batch_info = load_model_from_checkpoint(checkpoint_file)
        result = evaluate_model(model, tokenizer_path)
        checkpoint_name = os.path.basename(
            checkpoint_file
        )  # Extracting just the filename
        # get just the number
        checkpoint_name = extract_checkpoint_number(checkpoint_name)
        result[
            "checkpoint_name"
        ] = checkpoint_name  # Adding the checkpoint name to the checkpoint_output_results
        checkpoint_output_results.append(result)

        # Ensure checkpoint_name is of string type
        checkpoint_df = pl.DataFrame({"checkpoint_name": [checkpoint_name]})

        # Filter the DataFrame to only include the batch indices we trained on
        batch_info = list(
            chain(*[tensor.tolist() for tensor in batch_info])
        )  # flatten list of tensors

        df_filtered = df_metrics.filter(
            df_metrics["global_idx"].is_in(batch_info)
        )  # filter to only include batch indices we trained on

        # Compute the mean for each column
        batch_results = df_filtered.mean()
        # Horizontally concatenate batch_results and checkpoint_df
        batch_results = batch_results.hstack(checkpoint_df)

        checkpoint_batch_results = checkpoint_batch_results.vstack(batch_results)

    # Create a Polars DataFrame from checkpoint_output_results
    output_df = pl.DataFrame(checkpoint_output_results)
    output_df.write_csv("out/tables/summary.csv")
    # Create a Polars DataFrame from checkpoint_batch_results
    print(checkpoint_batch_results)
    checkpoint_batch_results.write_csv("out/tables/batch_results.csv")

    return print("Done!")


# -----------------------------------------------------------------------------
# run evaluation

if __name__ == "__main__":
    evaluate_all_checkpoints(checkpoint_dir, tokenizer_path, metrics_csv_path)
