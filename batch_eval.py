from tokenizer import Tokenizer
import polars as pl
from evaluators import evaluate_textual_metrics
from tinystories import Task
from train import max_seq_len, vocab_size, vocab_source


tokenizer_path = "tokenizer.model"
expected_stdout = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"


# Evaluate a batch of text against the expected output
def evaluate_batch(x, y, tokenizer_path):
    enc = Tokenizer(tokenizer_model=tokenizer_path)
    batch_results = []

    for idx in range(x.size(0)):  # Assuming x is a batch of sequences
        sequence = x[idx].tolist()
        text = enc.decode(sequence)
        metrics = evaluate_textual_metrics(text, expected_stdout.decode("utf-8"))
        metrics["text"] = text
        batch_results.append(metrics)

    return batch_results


# Main evaluation function
def evaluate_all_batches(batch_size, device, tokenizer_path, **dataset_kwargs):
    results = []
    batch_count = 0  # Initialize the batch count

    for x, y in Task.iter_batches(batch_size, device, **dataset_kwargs):
        if batch_count == 10:  # Break after 10 batches
            break

        print(f"Evaluating batch {len(results)//batch_size + 1}")
        batch_results = evaluate_batch(x, y, tokenizer_path)
        results.extend(batch_results)

        batch_count += 1  # Increment the batch count

    # Create a Polars DataFrame from results
    df = pl.DataFrame(results)

    # Save results to CSV using Polars
    df.write_csv(f"out/tables/{split}_batch_summary.csv")  # Saving split-wise

    return results


# Run evaluation
if __name__ == "__main__":
    batch_size = 16  # Adjust as needed
    device = "cpu"  # Adjust as needed: "cuda" or "cpu"
    dataset_kwargs = {}  # Provide necessary dataset arguments here

    for split in ["train", "val", "test"]:
        dataset_kwargs[
            "split"
        ] = split  # Assuming you might need the split in dataset_kwargs
        evaluate_all_batches(batch_size, device, tokenizer_path, **dataset_kwargs)
        print(f"Done with {split}!")
