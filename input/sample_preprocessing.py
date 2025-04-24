import pickle
import json
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from CSV
df = pd.read_csv('input/Iris.csv')

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Extract features and labels
x = df.drop('Species', axis=1)
y = df['Species']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Define the KNN classifier
classifier = neighbors.KNeighborsClassifier()

# Define the parameter grid for tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Use GridSearchCV for hyperparameter tuning with 5-fold cross-validation
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Best model and parameters
best_classifier = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions
predictions = best_classifier.predict(x_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True)

# Save the best model
with open('best_knn_model.pkl', 'wb') as model_file:
    pickle.dump(best_classifier, model_file)

# Save the evaluation results to a JSON file
evaluation_results = {
    'best_parameters': best_params,
    'accuracy_score': accuracy,
    'classification_report': report
}

with open('best_knn_evaluation.json', 'w') as json_file:
    json.dump(evaluation_results, json_file, indent=4)

print("Best model saved to 'best_knn_model.pkl'")
print("Evaluation results saved to 'best_knn_evaluation.json'")
