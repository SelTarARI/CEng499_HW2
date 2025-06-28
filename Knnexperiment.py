import sys
import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import sem  # Standard error of the mean for confidence intervals

# Add the directory containing Distance.py to the Python path
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from Distance import Distance
from Part1.Knn import KNN



# Load dataset and labels
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))

# Hyperparameter configurations
k_values = [1, 3, 5, 7, 9]
distance_metrics = {
    "Cosine": Distance.calculateCosineDistance,
    "Minkowski": Distance.calculateMinkowskiDistance,
    "Mahalanobis": Distance.calculateMahalanobisDistance,
}

# Precompute Mahalanobis inverse covariance matrix
inverse_covariance = np.linalg.inv(np.cov(dataset, rowvar=False))

# Stratified k-fold setup
n_splits = 10
n_repeats = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results
results = []

for shuffle_id in range(n_repeats):
    print(f"Shuffle {shuffle_id + 1}/{n_repeats}...")
    
    for k in k_values:
        for metric_name, metric_function in distance_metrics.items():
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            
            # Perform stratified 10-fold cross-validation
            for train_index, test_index in skf.split(dataset, labels):
                train_data, test_data = dataset[train_index], dataset[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]
                
                # Assign appropriate parameters for the distance metric
                if metric_name == "Minkowski":
                    params = 2  # Use p=2 (Euclidean distance)
                elif metric_name == "Mahalanobis":
                    params = inverse_covariance
                else:
                    params = None
                
                # Initialize and train the KNN model
                knn = KNN(train_data, train_labels, metric_function, params, K=k)
                
                # Predict and evaluate on the test set
                predictions = np.array([knn.predict(instance) for instance in test_data])
                
                # Calculate accuracy
                accuracy = np.mean(predictions == test_labels)
                accuracies.append(accuracy)
                
                # Calculate precision, recall, and F1-score
                precision = precision_score(test_labels, predictions, average='macro')
                recall = recall_score(test_labels, predictions, average='macro')
                f1 = f1_score(test_labels, predictions, average='macro')
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

            # Compute mean and confidence interval for accuracy and other metrics
            mean_accuracy = np.mean(accuracies)
            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)
            mean_f1 = np.mean(f1_scores)
            confidence_interval = 1.96 * sem(accuracies)  # 95% CI for accuracy

            # Store the results for this configuration
            results.append((k, metric_name, mean_accuracy, mean_precision, mean_recall, mean_f1, confidence_interval))

# Convert results to NumPy array for better formatting
results_array = np.array(results, dtype=object)

# Print the results as a table
header = ["K", "Metric", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1 Score", "95% Confidence Interval"]
print(f"{' | '.join(header)}")
print('-' * 100)

# Format each row and print it
for result in results_array:
    k, metric, mean_accuracy, mean_precision, mean_recall, mean_f1, confidence_interval = result
    print(f"{k:3} | {metric:10} | {mean_accuracy:.4f}  | {mean_precision:.4f}   | {mean_recall:.4f}    | {mean_f1:.4f}     | ±{confidence_interval:.4f}")

# Save results to a text file
with open("knn_results.txt", "w") as f:
    f.write(f"{' | '.join(header)}\n")
    f.write('-' * 100 + "\n")
    for result in results_array:
        k, metric, mean_accuracy, mean_precision, mean_recall, mean_f1, confidence_interval = result
        f.write(f"{k:3} | {metric:10} | {mean_accuracy:.4f}  | {mean_precision:.4f}   | {mean_recall:.4f}    | {mean_f1:.4f}     | ±{confidence_interval:.4f}\n")

print("Experiment completed. Results saved to 'knn_results.txt'.")
