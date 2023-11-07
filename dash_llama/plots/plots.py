import os
import torch
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px


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


##### VECTOR LAYER ANALYSIS #####

# single epoch


def plot_vectors(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "last_attn_weights" not in checkpoint:
        return None  # Handle error case appropriately in your application

    # Retrieve all attention weights and reshape them into a list of vectors, one per layer
    layers_vectors = [
        layer_weight.view(-1)
        .cpu()
        .numpy()  # Flatten each layer's weights into one vector
        for layer_weight in checkpoint["last_attn_weights"]
    ]

    # Normalize and perform PCA on the list of layer vectors
    normalized_vectors = np.array(
        [
            vector / np.linalg.norm(vector)
            for vector in layers_vectors
            if np.linalg.norm(vector) > 0
        ]
    )
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_vectors)

    # Compute cosine similarity matrix for the layer vectors
    cosine_sim_matrix = cosine_similarity(normalized_vectors)

    # Create a figure with two subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PCA Projections", "Cosine Similarity Heatmap"),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
    )

    # Add the PCA scatter plot to the first subplot
    fig.add_trace(
        go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode="markers+text",
            text=[f"Layer {i+1}" for i in range(len(normalized_vectors))],
            marker=dict(
                size=8,
                color=np.arange(len(normalized_vectors)),
                colorscale="Viridis",
                showscale=True,
            ),
            name="PCA",
        ),
        row=1,
        col=1,
    )

    # Add the cosine similarity heatmap to the second subplot
    fig.add_trace(
        go.Heatmap(
            z=cosine_sim_matrix,
            colorscale="Cividis",
            colorbar=dict(title="Cosine Similarity", x=1.1),
            name="Cosine Similarity",
        ),
        row=1,
        col=2,
    )

    # Update the layout of the figure
    fig.update_layout(title="Layer Vector Analysis")

    return fig


# multiple epochs
def plot_vectors_multiple_epochs(epoch_paths):
    pca = PCA(n_components=2)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PCA Projections", "Cosine Similarity Across Layers"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        horizontal_spacing=0.1,  # Adjust spacing to your liking
    )

    # Define a color palette for epochs
    colorscale = px.colors.qualitative.Set1

    # Store PCA results for each epoch
    pca_results = []

    # Iterate over each epoch to calculate PCA projections
    for i, checkpoint_path in enumerate(epoch_paths):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if "last_attn_weights" not in checkpoint:
            continue

        # Flatten and normalize the attention weights
        layers_vectors = [
            layer_weight.view(-1).cpu().numpy()
            for layer_weight in checkpoint["last_attn_weights"]
        ]
        normalized_vectors = [
            vector / np.linalg.norm(vector)
            for vector in layers_vectors
            if np.linalg.norm(vector) > 0
        ]
        pca_result = pca.fit_transform(normalized_vectors)
        pca_results.append(pca_result)

        # Plot PCA projections with text labels for each epoch with a unique color
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode="markers+text",
                text=[f"{i+1}" for i in range(len(pca_result))],  # Layer numbers
                textposition="top center",
                marker=dict(size=10, color=colorscale[i % len(colorscale)]),
                name=f"Epoch {i+1}",
            ),
            row=1,
            col=1,
        )

    # Calculate cosine similarity for each layer across epochs and plot
    num_layers = len(normalized_vectors)
    if len(epoch_paths) > 1:
        # Initialize array to store average cosine similarity for each layer
        avg_cosine_sim = np.zeros(num_layers)

        # Sum cosine similarities between the same layers across all pairs of epochs
        for i in range(len(epoch_paths)):
            for j in range(i + 1, len(epoch_paths)):
                for layer_idx in range(num_layers):
                    avg_cosine_sim[layer_idx] += cosine_similarity(
                        [pca_results[i][layer_idx]], [pca_results[j][layer_idx]]
                    )[0][0]

        # Calculate the average cosine similarity for each layer
        avg_cosine_sim /= (len(epoch_paths) * (len(epoch_paths) - 1)) / 2

        # Plot the average cosine similarity for each layer
        fig.add_trace(
            go.Scatter(
                x=[f"Layer {i+1}" for i in range(num_layers)],
                y=avg_cosine_sim,
                mode="lines+markers",
                name="Average Cosine Similarity",
            ),
            row=1,
            col=2,
        )

        # Update layout
    fig.update_layout(
        title="Vector Analysis Across Epochs",
        # legend=dict(orientation="h", x=0.5, y=1.1, xanchor="center", yanchor="top"),
        xaxis_title="PCA Dimension 1",
        yaxis_title="PCA Dimension 2",
        xaxis2_title="Layer",
        yaxis2_title="Average Cosine Similarity",
    )

    # Update xaxis properties if needed
    fig.update_xaxes(title_text="Layer Number", row=1, col=1)

    # Update yaxis properties if needed
    fig.update_yaxes(title_text="Cosine Similarity", row=1, col=2)

    return fig


#### SAT CURVES ####


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
def plot_sat_curves_from_parquet(output_path, batch_path, title):
    # Load the Parquet files into Pandas DataFrames
    output_df = pd.read_parquet(output_path)
    batch_df = pd.read_parquet(batch_path)

    # sort both dataframes by checkpoint_name
    output_df.sort_values(by="checkpoint_name", inplace=True)
    batch_df.sort_values(by="checkpoint_name", inplace=True)

    # Assuming both DataFrames have the same structure and the same metrics
    metrics = [
        col
        for col in output_df.columns
        if col not in ["checkpoint_name", "text", "global_idx"]
    ]

    # Create a subplot figure with shared y-axes and a shared legend
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        shared_xaxes=False,
        subplot_titles=("Input Batch Benchmarks", "Output Ckpt Benchmarks"),
    )

    for i, metric in enumerate(metrics):
        normalized_output = normalize_data(output_df[metric].tolist())
        normalized_batch = normalize_data(batch_df[metric].tolist())
        hovertemplate = f"Metric: {metric}<br>Value: {{y:.3f}}<extra></extra>"

        # Add traces for output benchmarks
        fig.add_trace(
            go.Scatter(
                x=output_df["checkpoint_name"],
                y=normalized_output,
                mode="lines+markers",
                name=metric,
                hovertemplate=hovertemplate,
            ),
            row=1,
            col=2,
        )

        # Add traces for batch benchmarks
        fig.add_trace(
            go.Scatter(
                x=batch_df["checkpoint_name"],
                y=normalized_batch,
                mode="lines+markers",
                name=metric,
                hovertemplate=hovertemplate,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Normalized Metric Value",
        xaxis1_title="Checkpoint Number",
        xaxis2_title="Checkpoint Number",
        legend_title="Metric",
        hovermode="closest",
    )

    return fig
