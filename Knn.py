import numpy as np

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        """
        Predicts the label for a given data instance.
        :param instance: A single data point (1D numpy array).
        :return: Predicted label for the instance.
        """
        distances = []
        for i in range(len(self.dataset)):
            if self.similarity_function_parameters is not None:
                # Pass parameters if they exist
                distance = self.similarity_function(instance, self.dataset[i], self.similarity_function_parameters)
            else:
                # Call the function without parameters
                distance = self.similarity_function(instance, self.dataset[i])
            distances.append(distance)
        
        indices = np.argsort(distances)
        nearest_labels = self.dataset_label[indices[:self.K]]
        return np.random.choice(np.flatnonzero(np.bincount(nearest_labels) == np.max(np.bincount(nearest_labels))))

