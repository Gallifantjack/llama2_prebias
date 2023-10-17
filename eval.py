import os
import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path


checkpoint_dir = "out/ckpt/"
tokenizer_path = get_tokenizer_model_path(vocab_size=32000)

# -----------------------------------------------------------------------------
# test utilities
expected_stdout = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"


def load_model_from_checkpoint(checkpoint_path):
    device = "cpu"
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict["model_args"])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


# -----------------------------------------------------------------------------
# tests


def evaluate_model(model, tokenizer_path):
    x = torch.tensor([[1]], dtype=torch.long, device="cpu")  # 1 is BOS
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=200, temperature=0.0)
    pt_tokens = y[0].tolist()

    enc = Tokenizer(tokenizer_model=tokenizer_path)
    text = enc.decode(pt_tokens)
    text = text.encode("ascii")

    assert text == expected_stdout  # This is based on your old test


# -----------------------------------------------------------------------------
# run scripts
def evaluate_all_checkpoints(checkpoint_dir, tokenizer_path):
    checkpoint_files = [
        os.path.join(checkpoint_dir, file)
        for file in os.listdir(checkpoint_dir)
        if file.endswith(".bin")
    ]

    for idx, checkpoint_file in enumerate(checkpoint_files):
        print(f"Evaluating checkpoint {idx+1}/{len(checkpoint_files)}")
        model = load_model_from_checkpoint(checkpoint_file)
        evaluate_model(model, tokenizer_path)


evaluate_all_checkpoints(checkpoint_dir, tokenizer_path)
