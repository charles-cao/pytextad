from sklearn.neighbors import NearestNeighbors
import numpy as np

class TokenCore:
    """
    TokenCore anomaly detector based on nearest neighbor distances
    """
    def __init__(self, n_neighbors=1, aggregation='max'):
        """
        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use
        aggregation : str, default='mean'
            How to aggregate distances when n_neighbors > 1
            Options: 'mean', 'max', 'min', 'median'
        """
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation
        self.nbrs = None
        
    def fit(self, X):
        """
        Fit the model using X as training data
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        return self
    
    def decision_function(self, X):
        """
        Compute anomaly scores for samples in X
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        scores : array-like of shape (n_samples,)
            Anomaly scores (higher means more anomalous)
        """
        distances, _ = self.nbrs.kneighbors(X)
        
        # If n_neighbors == 1, distances is (n_samples, 1), just flatten
        if self.n_neighbors == 1:
            scores = distances.flatten()
        else:
            # Aggregate multiple neighbor distances
            if self.aggregation == 'mean':
                scores = np.mean(distances, axis=1)
            elif self.aggregation == 'max':
                scores = np.max(distances, axis=1)
            elif self.aggregation == 'min':
                scores = np.min(distances, axis=1)
            elif self.aggregation == 'median':
                scores = np.median(distances, axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return scores
