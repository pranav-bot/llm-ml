# Complete Training Pipeline with Improved Preprocessing and Model Evaluation

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor  # Keeping KNN for comparison
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Trying tree-based models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import joblib

def preprocess_data(df):
    # Separate features and target variable
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    # Identify numerical and categorical features
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Create numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Use median imputation for numerical features
        ('scaler', StandardScaler()) # Scale numerical features
    ])

    # Create categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Use most_frequent imputation for categorical features
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])
    
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y

# Load the dataset
df = pd.read_csv('input2/housing.csv')

# Split data using improved preprocessing
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter grids
models = {
    'KNN': (KNeighborsRegressor(), {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}),
    'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
    'GradientBoosting': (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]}) 
}

best_model = None
best_score = -np.inf

for name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5)
    grid_search.fit(X_train, y_train)
    
    y_pred = grid_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    if r2 > best_score:
        best_score = r2
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

print(f"Best Model: {best_model.__class__.__name__}")
print(f"Best Parameters: {best_params}")
print(f"Best R-squared: {best_score}")


# Old Preprocessing and Model Saving (using joblib for parallel processing)
# Load the dataset
df_old = pd.read_csv('input2/housing.csv')

# Old preprocessing (provided in sample_preprocessing.py)
# Assuming sample_preprocessing.py contains a function called preprocess_data_old(df)

# Save the current model using joblib
joblib.dump(best_model, 'best_regressor.pkl')

# Load the saved KNN model (old model)
old_model = joblib.load('input2/best_knn_regressor.pkl')

# Evaluate the old model on the test data (using improved preprocessing)
old_y_pred = old_model.predict(X_test) # Using X_test from *improved* preprocessing
old_r2 = r2_score(y_test, old_y_pred)

print(f"Old Model R-squared: {old_r2}")
print(f"Difference in R-squared: {best_score - old_r2}")