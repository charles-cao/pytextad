# Original implementation of SIK by Yang Cao, included unchanged below this header.
# Verified identical to tests/reference/sik_original.py (see tests/test_vector.py).
import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class SIK:
    
    def __init__(self, max_samples=16, n_estimators=200, novelty=True, sparse=False, device='cpu', random_state=None):
        """
        Boundary Kernel Isolation-based Nearest Neighbor Ensemble for outlier detection.
        
        Parameters
        ----------
        max_samples : int, default=16
            The number of samples to draw for each estimator.
        
        n_estimators : int, default=200
            The number of base estimators in the ensemble.
        
        novelty : bool, default=False
            If True, the model is used for novelty detection and must be fitted on a 
            dataset of normal samples. If False, the model is used for outlier detection 
            where training data may contain outliers.
            
        sparse : bool, default=False
            If True, use sparse matrix implementation for memory efficiency.
            If False, use dense matrix implementation.
            
        device : str, default='cpu'
            Device to use for computations. Options:
            - 'cpu': Use CPU for computations.
            - 'cuda': Use NVIDIA GPU if available.
            - 'mps': Use Apple Metal Performance Shaders if available.
            - 'auto': Automatically select the best available device.
            Note: When sparse=True, only 'cpu' is supported.
        
        random_state : int, RandomState instance or None, default=None
            Controls the random seed used for generating random sample indices.
        """
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.novelty = novelty
        self.sparse = sparse
        self.device_str = device.lower()
        self.random_state = random_state
        
        # Only use PyTorch if it's available and not using sparse matrices
        self.use_torch = HAS_TORCH and not sparse
        
        # Warn if attempting to use a device with sparse matrices
        if self.sparse and self.device_str != 'cpu':
            print("Warning: Sparse matrices only support CPU computation. Ignoring device parameter.")
        
        # Setup compute device
        self.device = None
        if self.use_torch:
            if self.device_str == 'auto':
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            elif self.device_str == 'cuda' and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif self.device_str == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif self.device_str == 'cpu':
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print(f"Warning: Device '{device}' not available. Using CPU instead.")
    
    def fit(self, data):
        """
        Fit the model using data as training samples.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to float32 only when using PyTorch for better GPU performance
        if self.use_torch and self.device is not None and self.device.type != 'cpu':
            self.data = np.asarray(data, dtype=np.float32)
        else:
            self.data = np.asarray(data)
            
        n_samples, n_features = self.data.shape

        self.centroid = []
        self.centroids_radius = []
        
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            subIndex = rnd.choice(n_samples, self.max_samples, replace=False)
            self.centroid.append(subIndex)
            
            tdata = self.data[subIndex, :]
            
            tt_dis = cdist(tdata, tdata)
            radius = [] 
            for r_idx in range(self.max_samples):
                r = tt_dis[r_idx].copy()
                r[r < 0] = 0 
                r = np.delete(r, r_idx) 
                radius.append(np.min(r))
            self.centroids_radius.append(radius)        
        
        # If not in novelty mode, compute KME
        if not self.novelty:
            self.X_transformed_train_ = self._transform_internal(self.data)
            
            if self.sparse:
                # For sparse implementation, compute mean and convert to array
                mean_values = self.X_transformed_train_.mean(axis=0)
                self.kme_ = np.asarray(mean_values).ravel()
            else:
                # For dense implementation, use np.average
                self.kme_ = np.average(self.X_transformed_train_, axis=0)
        
        return self
    
    def _transform_internal(self, newdata):
        """
        Internal transform method used by both fit and transform.
        
        Returns matrix where each element is 0 if the sample is inside the
        hypersphere, 1 if it's outside. Matrix type depends on self.sparse.
        """
        n, _ = newdata.shape
        
        if self.sparse:
            # Sparse implementation
            rows = []
            cols = []
            
            for i in range(self.n_estimators):
                subIndex = self.centroid[i]
                radius = self.centroids_radius[i]
                tdata = self.data[subIndex, :]
                
                dis = cdist(tdata, newdata)
                
                centerIdx = np.argmin(dis, axis=0)
                
                min_distances = dis[centerIdx, np.arange(n)]
                
                radii = np.array(radius)[centerIdx]
                
                # Find indices where points are outside the hypersphere
                outside_indices = np.where(min_distances > radii)[0]
                
                # Collect coordinates for non-zero elements
                if len(outside_indices) > 0:
                    rows.extend(outside_indices)
                    cols.extend([i] * len(outside_indices))
            
            # Create data array (all 1s for binary case)
            data = np.ones(len(rows), dtype=np.int8)
            
            # Create CSR matrix directly
            result = csr_matrix((data, (rows, cols)), shape=(n, self.n_estimators), dtype=np.int8)
        else:
            # Dense implementation
            result = np.zeros((n, self.n_estimators))
            
            for i in range(self.n_estimators):
                subIndex = self.centroid[i]
                radius = self.centroids_radius[i]
                tdata = self.data[subIndex, :]
                    
                # if self.use_torch and self.device is not None:
                #     # Use PyTorch for accelerated distance computation
                #     tdata_tensor = torch.tensor(tdata, device=self.device)
                #     newdata_tensor = torch.tensor(newdata, device=self.device)
                #     dis = torch.cdist(tdata_tensor, newdata_tensor).cpu().numpy()
                # else:
                #     # Use original cdist
                #     dis = cdist(tdata, newdata)

                if self.use_torch and self.device is not None and self.device.type != 'cpu':
                    # Use PyTorch for accelerated distance computation (GPU/MPS)
                    tdata_tensor = torch.tensor(tdata.astype(np.float32), device=self.device)
                    newdata_tensor = torch.tensor(newdata.astype(np.float32), device=self.device)
                    dis = torch.cdist(tdata_tensor, newdata_tensor).cpu().numpy()
                else:
                    # Use original cdist
                    dis = cdist(tdata, newdata)

                
                centerIdx = np.argmin(dis, axis=0)
                
                min_distances = dis[centerIdx, np.arange(n)]
                
                radii = np.array(radius)[centerIdx]
                
                # in ball: 0; out ball: 1
                result[:, i] = (min_distances > radii).astype(int)
        
        return result
    
    def transform(self, X):
        """
        Transform X into the ensemble feature space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        X_transformed : ndarray or scipy.sparse.csr_matrix of shape (n_samples, n_estimators)
            The transformed data. Each element is 0 if sample is inside the
            hypersphere, 1 if it's outside.
        """
        check_is_fitted(self, ["centroid", "centroids_radius"])
        
        # If novelty=False and X is the training data, return pre-computed result
        if not self.novelty and hasattr(self, 'X_transformed_train_') and np.array_equal(X, self.data):
            return self.X_transformed_train_
        
        return self._transform_internal(X)

    def decision_function(self, X):
        """
        Compute the anomaly score for each sample in X using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly scores of the input samples.
            Higher scores indicate higher likelihood of being anomalies.
        """
        check_is_fitted(self, ["centroid", "centroids_radius"])
        
        X_transformed = self.transform(X)
        
        if self.novelty:
            # For novelty detection (training data is all normal)
            if self.sparse:
                # For sparse: count non-zeros in each row
                row_sums = X_transformed.sum(axis=1)
                return np.asarray(row_sums).ravel()
            else:
                # For dense: sum along rows
                return np.sum(X_transformed, axis=1)
        else:
            # For outlier detection with KME
            if self.sparse:
                # For sparse: dot product is already optimized
                return X_transformed.dot(self.kme_)
            else:
                # For dense: use np.dot
                return np.dot(X_transformed, self.kme_.T)
