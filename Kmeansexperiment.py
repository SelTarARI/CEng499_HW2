import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add the directory containing Distance.py to the Python path
#sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# Load preprocessed datasets
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

# Function to perform clustering and plot results
def perform_kmeans(dataset, dataset_name, max_k=10):
    losses = []
    silhouette_scores = []
    K_values = range(2, max_k + 1)

    for K in K_values:
        # Apply KMeans
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dataset)

        # Compute metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(dataset, labels)

        losses.append(inertia)
        silhouette_scores.append(silhouette_avg)

    # Plot Elbow Method (Loss vs K)
    plt.figure(figsize=(10, 5))
    plt.plot(K_values, losses, marker='o', label='Inertia (Loss)')
    plt.title(f"KMeans Loss (Elbow Method) - {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot Silhouette Score vs K
    plt.figure(figsize=(10, 5))
    plt.plot(K_values, silhouette_scores, marker='o', label='Silhouette Score')
    plt.title(f"KMeans Silhouette Scores - {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.legend()
    plt.show()

    print(f"Optimal K for {dataset_name} (Elbow): Choose K based on the elbow in the loss curve.")
    print(f"Optimal K for {dataset_name} (Silhouette): Choose K with highest silhouette score.")

# Perform KMeans for both datasets
perform_kmeans(dataset1, "Dataset 1")
perform_kmeans(dataset2, "Dataset 2")
