randomForestParams = {
    "criterion": ["gini", "entropy"],  # Criterion for measuring the quality of a split
    "max_depth": ["continuous"],  # Maximum depth of the tree, should be a positive integer or None (which means no limit)
    "min_samples_split": ["continuous"],  # The minimum number of samples required to split an internal node
    "min_samples_leaf": ["continuous"],  # The minimum number of samples required at a leaf node
    "min_weight_fraction_leaf": ["continuous"],  # The minimum weighted fraction of the total input samples required to be at a leaf node
    "max_features": ["auto", "sqrt", "log2", None, "continuous"],  # The number of features to consider when looking for the best split
    "max_leaf_nodes": ["continuous"],  # Grow trees with a maximum number of leaf nodes
    "random_state": ["continuous"],  # Seed used by the random number generator (use integers for reproducibility)
    "min_impurity_decrease": ["continuous"],  # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    "bootstrap": [True, False],  # Whether bootstrap samples are used when building trees
    "oob_score": [True, False],  # Whether to use out-of-bag samples to estimate the generalization accuracy
    "n_jobs": ["continuous"],  # The number of jobs to run in parallel for both fit and predict
    "verbose": ["continuous"],  # Controls the verbosity when fitting and predicting
    "warm_start": [True, False],  # Whether to reuse the solution of the previous call to fit and add more estimators to the ensemble
    "class_weight": [None, "balanced"],  # Weights associated with classes in classification tasks (None or "balanced" for auto-adjusting class weights)
    "n_estimators": ["continuous"],  # The number of trees in the forest
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm used to compute the nearest neighbors
    "leaf_size": ["continuous"],  # Leaf size passed to ball_tree or kd_tree (controls how efficiently the nearest neighbor search is performed)
    "splitter": ["best", "random"],  # Splitting strategy used by decision trees in the forest
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],  # Solver for optimization in certain tree-based models
    "penalty": ["l1", "l2", "elasticnet", "none"],  # Regularization penalties for linear classifiers
    "multi_class": ["auto", "ovr", "multinomial"],  # Multi-class option for classifiers
    "loss": ["hinge", "log", "squared_loss", "perceptron"],  # Loss function used in certain models
    "learning_rate": ["optimal", "constant", "invscaling", "adaptive"],  # Learning rate options for boosting models
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],  # Kernel used in certain models (e.g., SVM or Kernel Ridge)
    "decision_function_shape": ["ovo", "ovr"],  # Decision function shape (use "ovo" or "ovr" for multi-class classification)
}

