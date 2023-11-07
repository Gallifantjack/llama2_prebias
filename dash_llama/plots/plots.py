import os
import torch
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


# Plotly function to visualize attention from checkpoint
def plot_attention_from_checkpoint(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "last_attn_weights" not in checkpoint:
        return None  # Handle error case appropriately in your application

    # Retrieve the attention weights
    all_attn_weights = checkpoint["last_attn_weights"]
    n_layers = len(all_attn_weights)
    n_heads = all_attn_weights[0].size(1)

    # Generate subplots for each attention head
    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[
            f"Layer {i+1} Head {j+1}" for i in range(n_layers) for j in range(n_heads)
        ],
    )

    for i, layer_attn_weights in enumerate(all_attn_weights):
        for j in range(n_heads):
            attn_weights_slice = layer_attn_weights[0, j].cpu().numpy()
            fig.add_trace(go.Heatmap(z=attn_weights_slice), row=i + 1, col=j + 1)

    fig.update_layout(
        height=400 * n_layers,
        width=300 * n_heads,
        title_text="Attention Weights Visualization",
    )
    return fig


# Plotly function to visualize embeddings from checkpoint
def plot_embeddings_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "tok_embeddings" not in checkpoint:
        return None  # Handle error case appropriately in your application

    tok_embeddings = checkpoint["tok_embeddings"].cpu().numpy()
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(tok_embeddings)

    fig = go.Figure(
        data=go.Scatter(
            x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], mode="markers"
        )
    )
    fig.update_layout(
        title="Token Embeddings Visualization",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
    )

    return fig


def plot_vectors(checkpoint_path, batch_parquet_file_path):
    # Load output results from a Parquet file
    output_results_df = pd.read_parquet(checkpoint_path)

    # Assuming the DataFrame 'output_results_df' has a column 'all_layers_last_attn_weights' that contains the necessary data
    # You may need to adjust the following line based on the actual structure of your DataFrame
    all_layers_activations = [
        np.stack(weights) for weights in output_results_df["last_attn_weights"]
    ]

    # Load batch results from a Parquet file
    batch_results_df = pd.read_parquet(batch_parquet_file_path)

    # Perform your analysis with output results and batch results here
    # For now, let's just continue with the PCA part assuming 'all_layers_activations' is a list of NumPy arrays
    pca = PCA(n_components=2)
    normalized_vectors = [
        activations / np.linalg.norm(activations, axis=1, keepdims=True)
        for activations in all_layers_activations
    ]
    flattened_data = np.concatenate(normalized_vectors, axis=0)

    pca_result = pca.fit_transform(flattened_data)
    cosine_sim_matrix = cosine_similarity(flattened_data)

    # Create two subplots, one for PCA, one for cosine similarity heatmap
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("PCA Projections", "Cosine Similarities")
    )

    # PCA Scatter plot
    fig.add_trace(
        go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode="markers",
            marker=dict(
                color=np.arange(len(pca_result)),
                colorscale="Viridis",
                colorbar=dict(title="Layer Index"),
            ),
        ),
        row=1,
        col=1,
    )

    # Cosine Similarity Heatmap
    fig.add_trace(go.Heatmap(z=cosine_sim_matrix, colorscale="Cividis"), row=1, col=2)

    fig.update_layout(title_text="Vector Analysis Visualization")
    return fig


def normalize_data(data_list):
    """Normalize data to [0, 1] range."""
    min_val = min(data_list)
    max_val = max(data_list)

    if min_val == max_val:
        return [0] * len(data_list)

    return [(x - min_val) / (max_val - min_val) for x in data_list]


def extract_checkpoint_number(filename):
    try:
        # Updated extraction based on the 'ckpt_NUMBER.pt' format
        return int(filename.split("ckpt_")[1].split(".pt")[0])
    except IndexError:
        print(f"Unexpected filename structure: {filename}")
        return None


# Function to plot SAT curves from a Parquet file using Plotly
def plot_sat_curves_from_parquet(file_path, title):
    df = pd.read_parquet(file_path)
    # df["checkpoint_number"] = df["checkpoint_name"].apply(extract_checkpoint_number)
    df.sort_values(by="checkpoint_name", inplace=True)

    excluded_columns = ["checkpoint_number", "text", "global_idx"]
    metrics = [col for col in df.columns if col not in excluded_columns]

    # Create the figure for plotting
    fig = go.Figure()

    for metric in metrics:
        normalized_values = normalize_data(df[metric].tolist())
        fig.add_trace(
            go.Scatter(
                x=df["checkpoint_name"],
                y=normalized_values,
                mode="lines+markers",
                name=metric,
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Checkpoint Number",
        yaxis_title="Normalized Metric Value",
        legend_title="Metric",
    )

    return fig
