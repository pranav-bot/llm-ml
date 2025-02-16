KNNParams = {
    "n_neighbors": ["continuous"],  # The number of neighbors to use for k-nearest neighbors
    "weights": ["uniform", "distance"],  # Weight function used in prediction ("uniform" for equal weights, "distance" for inverse distance weighting)
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm used to compute the nearest neighbors
    "leaf_size": ["continuous"],  # Leaf size for ball_tree and kd_tree
    "p": [1, 2],  # Power parameter for the Minkowski distance (1 for Manhattan distance, 2 for Euclidean distance)
    "metric": ["minkowski", "euclidean", "manhattan", "chebyshev", "cosine", "hamming", "jaccard"],  # The distance metric to use
    "metric_params": ["continuous"],  # Additional parameters for the chosen metric
    "n_jobs": ["continuous"],  # The number of jobs to run in parallel for both fit and predict
    "algorithm_params": ["continuous"],  # Parameters specific to the algorithm used (e.g., for "ball_tree" or "kd_tree")
    "radius": ["continuous"],  # The radius within which to search for neighbors (for radius-based neighbors)
    "outlier_label": ["continuous"],  # The label to use for outliers (if any)
    "distance_threshold": ["continuous"],  # Distance threshold to apply when performing nearest neighbor search
}
