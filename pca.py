import numpy as np

class PCA:
    def __init__(self, projection_dim: int):
        """
        Initializes the PCA method
        :param projection_dim: the projection space dimensionality
        """
        self.projection_dim = projection_dim
        self.projection_matrix = None  # To store the projection matrix

    def fit(self, x: np.ndarray) -> None:
        """
        Applies the PCA method and obtains the projection matrix
        :param x: the data matrix on which the PCA is applied
        :return: None
        """
        # Step 1: Center the data (subtract the mean)
        mean = np.mean(x, axis=0)
        centered_data = x - mean

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top 'projection_dim' eigenvectors
        self.projection_matrix = eigenvectors[:, :self.projection_dim]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Projects data onto the learned lower-dimensional space
        :param x: data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        """
        if self.projection_matrix is None:
            raise ValueError("The PCA projection matrix has not been computed. Call fit() first.")
        
        # Center the data (subtract the mean used during fit)
        mean = np.mean(x, axis=0)
        centered_data = x - mean

        # Apply the projection matrix
        return np.dot(centered_data, self.projection_matrix)
