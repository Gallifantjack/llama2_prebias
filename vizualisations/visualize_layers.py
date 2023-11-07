import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def analyze_vectors(checkpoint_path, viz_path, checkpoint_dir_name, idx):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "all_layers_last_attn_weights" not in checkpoint:
        print("No layer activations found in the checkpoint.")
        return

    all_layers_activations = [
        torch.stack(layer_activations)
        for layer_activations in checkpoint["all_layers_last_attn_weights"]
    ]

    pca = PCA(n_components=2)
    normalized_vectors = [
        activation / torch.norm(activation, dim=1, keepdim=True)
        for activation in all_layers_activations
    ]
    flattened_data = torch.cat(normalized_vectors, dim=0).numpy()

    pca_result = pca.fit_transform(flattened_data)
    cosine_sim_matrix = cosine_similarity(flattened_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_matrix, annot=False, cmap="coolwarm")
    cosine_similarity_save_path = get_save_path(
        viz_path, "cosine_similarities", checkpoint_dir_name, idx
    )
    plt.title(f"Cosine Similarities (Checkpoint {idx})")
    plt.savefig(cosine_similarity_save_path)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(
        pca_result[:, 0], pca_result[:, 1], c=np.arange(len(pca_result)), cmap="viridis"
    )
    plt.colorbar().set_label("Index in Flattened Layer Activations")
    plt.title(f"PCA Projections (Checkpoint {idx})")
    pca_save_path = get_save_path(viz_path, "pca_layers", checkpoint_dir_name, idx)
    plt.savefig(pca_save_path)
    plt.close()


def get_save_path(base_path, analysis_type, dir_name, index):
    path = os.path.join(base_path, analysis_type, dir_name, f"checkpoint_{index}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
