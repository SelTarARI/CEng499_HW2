import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import torch

# Add the directory containing Distance.py to the Python path
#sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from pca import PCA
from autoencoder import AutoEncoder

# Load datasets
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

def visualize(data, labels, title, method_name):
    """
    Utility function to visualize 2D data using scatter plots.
    :param data: 2D array (N x 2)
    :param labels: Labels or indices for coloring (optional)
    :param title: Plot title
    :param method_name: Dimensionality reduction method
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar()
    plt.title(f"{method_name} - {title}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def apply_dimensionality_reduction(dataset, dataset_name):
    """
    Applies PCA, t-SNE, UMAP, and Autoencoder on a dataset and visualizes the results.
    :param dataset: Input dataset (N x D)
    :param dataset_name: Name of the dataset for labeling plots
    """
    print(f"Processing {dataset_name}...")

    # PCA
    pca = PCA(projection_dim=2)
    pca.fit(dataset)
    pca_result = pca.transform(dataset)
    visualize(pca_result, None, dataset_name, "PCA")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(dataset)
    visualize(tsne_result, None, dataset_name, "t-SNE")

    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_reducer.fit_transform(dataset)
    visualize(umap_result, None, dataset_name, "UMAP")

    # Autoencoder
    dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
    autoencoder = AutoEncoder(input_dim=dataset.shape[1], projection_dim=2, learning_rate=0.01, iteration_count=100)
    autoencoder.fit(dataset_tensor)
    autoencoder_result = autoencoder.transform(dataset_tensor).detach().numpy()
    visualize(autoencoder_result, None, dataset_name, "Autoencoder")

# Apply dimensionality reduction on both datasets
apply_dimensionality_reduction(dataset1, "Dataset 1")
apply_dimensionality_reduction(dataset2, "Dataset 2")
