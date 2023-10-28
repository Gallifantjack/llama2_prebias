import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import polars as pl
import argparse

from vizualisations.visualize_attn import visualize_attention_from_checkpoint
from vizualisations.visualize_embd import visualize_embeddings_from_checkpoint
from vizualisations.visualize_sat_curves import plot_metrics_from_parquet


# Centralized main function to run after training loops
def visualize_all(out_dir):
    # paths
    viz_path = os.path.join(out_dir, "visualizations")
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

    checkpoint_dir = os.path.join(out_dir, "ckpt")
    checkpoint_dir_name = os.path.basename(os.path.normpath(checkpoint_dir))

    checkpoints = [
        os.path.join(checkpoint_dir, file)
        for file in os.listdir(checkpoint_dir)
        if file.endswith(".pt")
    ]

    # Visualizing Attention
    for idx, checkpoint_path in enumerate(checkpoints):
        print(f"Visualizing Attention for checkpoint {idx+1}/{len(checkpoints)}")
        attn_path = os.path.join(
            viz_path,
            "attn",
            checkpoint_dir_name,
            f"checkpoint_{idx+1}.png",
        )
        if not os.path.exists(os.path.dirname(attn_path)):
            os.makedirs(os.path.dirname(attn_path))
        visualize_attention_from_checkpoint(checkpoint_path, attn_path)

    # Visualizing Embeddings
    for idx, checkpoint_path in enumerate(checkpoints):
        print(f"Visualizing Embeddings for checkpoint {idx+1}/{len(checkpoints)}")
        embd_path = os.path.join(
            viz_path,
            "embd",
            checkpoint_dir_name,
            f"checkpoint_{idx+1}.png",
        )
        if not os.path.exists(os.path.dirname(embd_path)):
            os.makedirs(os.path.dirname(embd_path))
        visualize_embeddings_from_checkpoint(checkpoint_path, embd_path)

    # Visualizing SAT Curves
    print("Visualizing SAT Curves")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    # plot for output benchmarks
    checkpoint_output_results_path = os.path.join(
        out_dir, "metadata", "checkpoint_output_results.parquet"
    )
    plot_metrics_from_parquet(ax1, checkpoint_output_results_path, "sat_curves_output")
    # plot for batch benchmarks
    batch_parquet_file_path = os.path.join(out_dir, "metadata", "batch_results.parquet")
    plot_metrics_from_parquet(ax2, batch_parquet_file_path, "sat_curves_batch")
    # save the figure
    plt.tight_layout()
    sat_save_path = os.path.join(
        viz_path,
        "sat",
        checkpoint_dir_name,
        "batch_output_curves.png",
    )
    if not os.path.exists(os.path.dirname(sat_save_path)):
        os.makedirs(os.path.dirname(sat_save_path))
    plt.savefig(sat_save_path)
    plt.close()
