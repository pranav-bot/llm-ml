import pickle
import json
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the California housing dataset from CSV
df = pd.read_csv('input2/housing.csv')

# One-hot encode the 'ocean_proximity' categorical feature
df = pd.get_dummies(df, columns=['ocean_proximity'])

# Separate features and target
x = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Handle missing values using SimpleImputer (mean imputation)
imputer = SimpleImputer(strategy='mean')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Define the KNN Regressor
regressor = neighbors.KNeighborsRegressor()

# Define the parameter grid for tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Use GridSearchCV for hyperparameter tuning with 5-fold cross-validation
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

# Best model and parameters
best_regressor = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions
predictions = best_regressor.predict(x_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save the best model
with open('best_knn_regressor.pkl', 'wb') as model_file:
    pickle.dump(best_regressor, model_file)

# Save the evaluation results to a JSON file
evaluation_results = {
    'best_parameters': best_params,
    'mean_absolute_error': mae,
    'mean_squared_error': mse,
    'r2_score': r2
}

with open('best_knn_regression_evaluation.json', 'w') as json_file:
    json.dump(evaluation_results, json_file, indent=4)

print("Best regression model saved to 'best_knn_regressor.pkl'")
print("Evaluation results saved to 'best_knn_regression_evaluation.json'")
