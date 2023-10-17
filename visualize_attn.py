import os
import torch
import matplotlib.pyplot as plt


def visualize_attention_from_checkpoint(checkpoint_path, save_path):
    """
    Load the checkpoint, extract attention weights, and visualize them.

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        save_path (str): Path to save the generated visualization.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Check if the attention weights are in the checkpoint
    if "last_attn_weights" not in checkpoint:
        print("No attention weights found in the checkpoint.")
        return

    # Retrieve the attention weights
    all_attn_weights = checkpoint["last_attn_weights"]

    # Count the number of layers and heads
    n_layers = len(all_attn_weights)
    n_heads = all_attn_weights[0].size(
        1
    )  # assuming all layers have same number of heads

    # Create a new figure
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(15, 15))
    for i, layer_attn_weights in enumerate(all_attn_weights):
        for j in range(n_heads):
            ax = axes[i][j]
            attn_weights_slice = (
                layer_attn_weights[0, j].cpu().numpy()
            )  # first sample, j-th head
            ax.imshow(attn_weights_slice, cmap="viridis")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Head {j+1}")
            if j == 0:
                ax.set_ylabel(f"Layer {i+1}", rotation=90, va="center")

    plt.tight_layout()
    plt.savefig(save_path)


# Example usage
checkpoint_dir = "out/ckpt/"
checkpoints = [
    os.path.join(checkpoint_dir, file)
    for file in os.listdir(checkpoint_dir)
    if file.endswith(".pt")
]

for idx, checkpoint_path in enumerate(checkpoints):
    print(f"Processing checkpoint {idx+1}/{len(checkpoints)}")
    save_path = f"out/visualize/attn/checkpoint_{idx+1}.png"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    visualize_attention_from_checkpoint(checkpoint_path, save_path)
