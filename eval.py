import os
import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import polars as pl
from evaluators import evaluate_textual_metrics

checkpoint_dir = "out/ckpt/"
tokenizer_path = "tokenizer.model"
expected_stdout = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"

# -----------------------------------------------------------------------------
# test utilities


def load_model_from_checkpoint(checkpoint_file):
    checkpoint_dict = torch.load(checkpoint_file, map_location="cpu")
    model_args = ModelArgs(
        **checkpoint_dict["model_args"]
    )  # Adjusted this to match the test script
    model = Transformer(model_args)

    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    return model


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


def evaluate_all_checkpoints(checkpoint_dir, tokenizer_path):
    checkpoint_files = [
        os.path.join(checkpoint_dir, file)
        for file in os.listdir(checkpoint_dir)
        if file.endswith(".pt")
    ]

    results = []

    for idx, checkpoint_file in enumerate(checkpoint_files):
        print(f"Evaluating checkpoint {idx+1}/{len(checkpoint_files)}")
        model = load_model_from_checkpoint(checkpoint_file)
        result = evaluate_model(model, tokenizer_path)
        checkpoint_name = os.path.basename(
            checkpoint_file
        )  # Extracting just the filename
        result[
            "checkpoint_name"
        ] = checkpoint_name  # Adding the checkpoint name to the results
        results.append(result)

    # Create a Polars DataFrame from results
    df = pl.DataFrame(results)

    # Save results to CSV using Polars
    df.write_csv("out/tables/summary.csv")

    return results


# -----------------------------------------------------------------------------
# run evaluation

if __name__ == "__main__":
    evaluate_all_checkpoints(checkpoint_dir, tokenizer_path)
    print("Done!")
