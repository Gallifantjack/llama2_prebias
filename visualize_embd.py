import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_embeddings_from_checkpoint(checkpoint_path, save_path):
    """
    Load the checkpoint, extract token embeddings, and visualize them using PCA.

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        save_path (str): Path to save the generated visualization.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Check if the token embeddings are in the checkpoint
    if "tok_embeddings" not in checkpoint:
        print("No token embeddings found in the checkpoint.")
        return

    # Retrieve the token embeddings
    tok_embeddings = checkpoint["tok_embeddings"].cpu().numpy()

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(tok_embeddings)

    # Plot the reduced embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Token Embeddings Visualization")

    # Save the figure
    plt.savefig(save_path)


# # Example usage
# checkpoint_dir = "out/ckpt/"
# checkpoints = [
#     os.path.join(checkpoint_dir, file)
#     for file in os.listdir(checkpoint_dir)
#     if file.endswith(".pt")
# ]

# for idx, checkpoint_path in enumerate(checkpoints):
#     print(f"Processing checkpoint {idx+1}/{len(checkpoints)}")
#     save_path = f"out/visualize/embd/checkpoint_{idx+1}.png"
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     visualize_embeddings_from_checkpoint(checkpoint_path, save_path)
