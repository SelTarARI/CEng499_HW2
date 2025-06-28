import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# Add the directory containing Distance.py to the Python path
#sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# Load preprocessed datasets
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

# Function to perform clustering and plot results
def perform_kmedoids(dataset, dataset_name, max_k=10, metric='euclidean'):
    losses = []
    silhouette_scores = []
    K_values = range(2, max_k + 1)

    for K in K_values:
        # Apply KMedoids
        kmedoids = KMedoids(n_clusters=K, metric=metric, random_state=42, init='k-medoids++')
        labels = kmedoids.fit_predict(dataset)

        # Compute metrics
        loss = kmedoids.inertia_
        silhouette_avg = silhouette_score(dataset, labels)

        losses.append(loss)
        silhouette_scores.append(silhouette_avg)

    # Plot Elbow Method (Loss vs K)
    plt.figure(figsize=(10, 5))
    plt.plot(K_values, losses, marker='o', label='Loss (Inertia)')
    plt.title(f"KMedoids Loss (Elbow Method) - {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot Silhouette Score vs K
    plt.figure(figsize=(10, 5))
    plt.plot(K_values, silhouette_scores, marker='o', label='Silhouette Score')
    plt.title(f"KMedoids Silhouette Scores - {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.legend()
    plt.show()

    print(f"Optimal K for {dataset_name} (Elbow): Choose K based on the elbow in the loss curve.")
    print(f"Optimal K for {dataset_name} (Silhouette): Choose K with highest silhouette score.")

# Perform KMedoids for both datasets
perform_kmedoids(dataset1, "Dataset 1")
perform_kmedoids(dataset2, "Dataset 2")
