import os
import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import polars as pl


from nltk.translate.bleu_score import sentence_bleu

checkpoint_dir = "out/ckpt/"

# Set the path to the tokenizer model
tokenizer_path = "tokenizer.model"

# -----------------------------------------------------------------------------
# test utilities
expected_stdout = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"


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


def bleu_evaluation(generated_text):
    reference = [expected_stdout.decode("ascii").split()]
    candidate = generated_text.split()
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score


def prevalence_of_word(generated_text, word):
    return generated_text.count(word) / len(generated_text.split())


def sentence_length(generated_text):
    return len(generated_text.split())


# -----------------------------------------------------------------------------
# test function


def evaluate_model(model, tokenizer_path):
    x = torch.tensor([[1]], dtype=torch.long, device="cpu")  # 1 is BOS
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=200, temperature=0.0)
    pt_tokens = y[0].tolist()
    enc = Tokenizer(tokenizer_model=tokenizer_path)
    generated_text = enc.decode(pt_tokens)

    # Evaluation functions
    bleu = bleu_evaluation(generated_text)
    prevalence_the = prevalence_of_word(generated_text, "the")
    prevalence_and = prevalence_of_word(generated_text, "and")
    sentence_len = sentence_length(generated_text)

    return {
        "text": generated_text,
        "bleu": bleu,
        "prevalence_the": prevalence_the,
        "prevalence_and": prevalence_and,
        "sentence_length": sentence_len,
    }


# -----------------------------------------------------------------------------
# run evaluation


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
        results.append(result)

    # Create a Polars DataFrame from results
    df = pl.DataFrame(results)

    # Save results to CSV using Polars
    df.write_csv("out/tables/summary.csv")

    return results


evaluate_all_checkpoints(checkpoint_dir, tokenizer_path)
