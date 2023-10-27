import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import polars as pl
import argparse

from visualize_attn import visualize_attention_from_checkpoint
from visualize_embd import visualize_embeddings_from_checkpoint
from visualize_sat_curves import plot_metrics_from_csv


# Centralized main function to run after training loops
def visualize_all(checkpoint_dir):
    # Determine directory name for the checkpoint for output structure
    checkpoint_dir_name = os.path.basename(os.path.normpath(checkpoint_dir))

    checkpoints = [
        os.path.join(checkpoint_dir, file)
        for file in os.listdir(checkpoint_dir)
        if file.endswith(".pt")
    ]

    # Visualizing Attention
    for idx, checkpoint_path in enumerate(checkpoints):
        print(f"Visualizing Attention for checkpoint {idx+1}/{len(checkpoints)}")
        save_path = f"out/visualize/attn/{checkpoint_dir_name}/checkpoint_{idx+1}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        visualize_attention_from_checkpoint(checkpoint_path, save_path)

    # Visualizing Embeddings
    for idx, checkpoint_path in enumerate(checkpoints):
        print(f"Visualizing Embeddings for checkpoint {idx+1}/{len(checkpoints)}")
        save_path = f"out/visualize/embd/{checkpoint_dir_name}/checkpoint_{idx+1}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        visualize_embeddings_from_checkpoint(checkpoint_path, save_path)

    # Visualizing SAT Curves
    print("Visualizing SAT Curves")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    output_csv_file_path = "out/tables/summary.csv"
    plot_metrics_from_csv(ax1, output_csv_file_path, "sat_curves_output")
    batch_csv_file_path = "out/tables/batch_results.csv"
    plot_metrics_from_csv(ax2, batch_csv_file_path, "sat_curves_batch")
    plt.tight_layout()
    plt.savefig(f"out/visualize/sat/{checkpoint_dir_name}/batch_output_curves.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize all aspects after training loops."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="out/ckpt/",
        help="Path to the directory containing model checkpoints.",
    )

    args = parser.parse_args()
    visualize_all(args.checkpoint_dir)
