import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # Alternative classifier
from sklearn.linear_model import LogisticRegression # Alternative classifier
from sklearn.ensemble import RandomForestClassifier # Alternative classifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings

# --- Configuration ---
DATA_PATH = 'input/Iris.csv'
OLD_MODEL_SAVE_PATH = 'output/old_knn_model.pkl'
IMPROVED_MODEL_SAVE_PATH = 'output/improved_best_model.pkl'
SCALER_SAVE_PATH = 'output/iris_scaler.pkl'
ENCODER_SAVE_PATH = 'output/iris_label_encoder.pkl'
OUTPUT_DIR = 'output'
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Create input directory if it doesn't exist (for dummy file creation)
os.makedirs('input', exist_ok=True)


# Ignore specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- Create Dummy Iris Dataset ---
# This section creates input/Iris.csv because the actual file is not accessible.
# In a real environment where input/Iris.csv exists, this block can be removed.
def create_dummy_iris(file_path):
    """Creates a dummy Iris.csv file for testing purposes."""
    print(f'Creating dummy dataset at {file_path}...')
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    df_iris = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                           columns=iris_data['feature_names'] + ['target'])
    # Map target numbers to species names to match the expected format
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    df_iris['Species'] = df_iris['target'].map(species_map)
    df_iris = df_iris.drop('target', axis=1)
    # Rename columns slightly to remove spaces/parentheses for easier handling
    df_iris.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'Species']
    # Add an 'Id' column just to test the drop logic
    df_iris.insert(0, 'Id', range(1, 1 + len(df_iris)))
    df_iris.to_csv(file_path, index=False)
    print(f'Dummy dataset created successfully.')

if not os.path.exists(DATA_PATH):
    create_dummy_iris(DATA_PATH)

# --- Placeholder Old Preprocessing ---
# Simulates older preprocessing: No scaling, no stratification
def old_preprocess_data(file_path, test_size=0.3, random_state=42):
    """
    Loads and preprocesses data using a basic, 'old' approach.
    - Loads data from CSV.
    - Drops 'Id' column.
    - Encodes 'Species' label.
    - Performs train-test split WITHOUT stratification.
    - Does NOT scale features.
    """
    print("\n--- Running Old Preprocessing ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None, None, None, None, None

    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("Dropped 'Id' column (Old Preprocessing).")

    try:
        X = df.drop('Species', axis=1)
        y_raw = df['Species']
    except KeyError:
        print("Error: 'Species' column not found (Old Preprocessing).")
        return None, None, None, None, None

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Encoded target variable 'Species'. Classes: {encoder.classes_} (Old Preprocessing)")

    # Simple split, no stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data (no stratification). Train: {len(X_train)}, Test: {len(X_test)}")
    print("Old preprocessing complete (no scaling).")
    # Return features as pandas DataFrames/Series as they were split
    return X_train, X_test, y_train, y_test, encoder

# --- Improved Preprocessing (From User Input) ---
def improved_preprocess_data(file_path, test_size=0.3, random_state=42, output_dir='output'):
    """
    Loads and preprocesses the Iris dataset using an improved approach.
    - Loads data from CSV.
    - Drops 'Id' column.
    - Encodes 'Species' label.
    - Performs STRATIFIED train-test split.
    - Applies StandardScaler (fitted on train, applied to train/test).
    - Saves scaler and encoder objects.
    """
    print("\n--- Running Improved Preprocessing ---")
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None, None, None, None, None, None

    # --- 2. Drop Unnecessary Columns ---
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("Dropped 'Id' column.")

    # --- 3. Separate Features and Target ---
    try:
        X = df.drop('Species', axis=1)
        y_raw = df['Species']
    except KeyError:
        print("Error: 'Species' column not found in the dataset.")
        return None, None, None, None, None, None

    # --- 4. Encode Target Variable ---
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Encoded target variable 'Species'. Classes: {encoder.classes_}")

    # --- 5. Split Data (Stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Crucial for maintaining class distribution
    )
    print(f"Split data into training ({len(X_train)} samples) and testing ({len(X_test)} samples) with stratification.")

    # --- 6. Initialize Scaler ---
    scaler = StandardScaler()

    # --- 7. Fit Scaler on Training Data ---
    # Fit *only* on the training data to avoid data leakage
    scaler.fit(X_train)
    print("Fitted StandardScaler on training data.")

    # --- 8. Transform Data ---
    # Apply the *same* fitted scaler to both train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Applied scaling transformation to training and testing features.")

    # --- 9. Save Scaler and Encoder (Optional but good practice) ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        scaler_path = os.path.join(output_dir, SCALER_SAVE_PATH)
        encoder_path = os.path.join(output_dir, ENCODER_SAVE_PATH)
        try:
            joblib.dump(scaler, scaler_path)
            joblib.dump(encoder, encoder_path)
            print(f"Saved scaler to {scaler_path}")
            print(f"Saved encoder to {encoder_path}")
        except Exception as e:
            print(f"Warning: Could not save scaler/encoder: {e}")


    print("Improved preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder

# --- Model Training and Evaluation Function ---
def train_evaluate(model, X_train, y_train, X_test, y_test, encoder, model_name="Model"):
    """Trains a model and returns evaluation metrics."""
    print(f"\n--- Training and Evaluating {model_name} ---")
    # Ensure input data is numpy array for consistency
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    if isinstance(X_test, pd.DataFrame): X_test = X_test.values
    if isinstance(y_train, pd.Series): y_train = y_train.values
    if isinstance(y_test, pd.Series): y_test = y_test.values

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    try:
      # Use the passed encoder's classes
      target_names = [str(cls) for cls in encoder.classes_]
      report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
      report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    except Exception as e:
       print(f"Warning: Could not get target names for report: {e}")
       report_str = classification_report(y_test, y_pred, zero_division=0)
       report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n{report_str}")

    # Extract macro f1-score safely
    macro_f1 = report_dict.get('macro avg', {}).get('f1-score', 0.0)

    return {'accuracy': accuracy, 'report': report_dict, 'macro_f1': macro_f1}


# --- Find Best Classifier with Improved Preprocessing (using GridSearchCV) ---
def find_best_model(X_train, y_train, X_test, y_test, encoder):
    """Finds the best classifier using GridSearchCV."""
    print("\n--- Finding Best Classifier with Improved Preprocessing using GridSearchCV ---")
    models_params = {
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'svm': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'], # Removed 'poly' as it can be slow/complex
                'gamma': ['scale', 'auto']
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(max_iter=200, multi_class='auto', random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        }
    }

    best_score = -1
    best_model_info = None

    # Use StratifiedKFold for classification CV
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, mp in models_params.items():
        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=cv_stratified, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best score for {name}: {grid_search.best_score_:.4f}")
        print(f"Best params for {name}: {grid_search.best_params_}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model_info = {
                'name': name,
                'model': grid_search.best_estimator_,
                'params': grid_search.best_params_,
                'cv_score': grid_search.best_score_
            }

    print(f"\n--- Best Overall Model Found: {best_model_info['name']} ---")
    print(f"Best CV Accuracy: {best_model_info['cv_score']:.4f}")
    print(f"Best Parameters: {best_model_info['params']}")

    # Evaluate the final best model on the test set
    best_model_test_metrics = train_evaluate(
        best_model_info['model'], X_train, y_train, X_test, y_test, encoder, f"Best Overall Model ({best_model_info['name']})"
    )
    best_model_info['test_metrics'] = best_model_test_metrics

    return best_model_info


# --- Main Execution ---
metrics_imp = None
metrics_old = None
best_model_info_imp = {'name': 'N/A'} # Default value

# 1. Improved Preprocessing and Best Model Training
imp_results = improved_preprocess_data(DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE, output_dir=OUTPUT_DIR)
if imp_results[0] is not None:
    X_train_imp, X_test_imp, y_train_imp, y_test_imp, scaler_imp, encoder_imp = imp_results

    # Find and evaluate the best model using improved data
    best_model_info_imp = find_best_model(X_train_imp, y_train_imp, X_test_imp, y_test_imp, encoder_imp)
    best_model_imp = best_model_info_imp['model']
    metrics_imp = best_model_info_imp['test_metrics']

    # Save the best *improved* model
    try:
        joblib.dump(best_model_imp, IMPROVED_MODEL_SAVE_PATH)
        print(f"\nSaved best model ({best_model_info_imp['name']}) with improved preprocessing to {IMPROVED_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save improved model: {e}")

else:
    print("\nImproved preprocessing failed. Cannot train improved model.")

# 2. Old Preprocessing and KNN Model Training (as specified)
old_results = old_preprocess_data(DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE)
if old_results[0] is not None:
    X_train_old, X_test_old, y_train_old, y_test_old, encoder_old = old_results

    # Train a standard KNN model (using params from original 'best_parameters')
    # Original metrics: {'algorithm': 'auto', 'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}
    old_knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='uniform')

    # Evaluate the old KNN model
    # Pass the correct encoder associated with this preprocessing step
    metrics_old = train_evaluate(old_knn_model, X_train_old, y_train_old, X_test_old, y_test_old, encoder_old, "Old KNN Model (Old Preprocessing)")

    # Save the model trained with old preprocessing
    try:
        joblib.dump(old_knn_model, OLD_MODEL_SAVE_PATH)
        print(f"\nSaved KNN model trained with old preprocessing to {OLD_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save old KNN model: {e}")

else:
    print("\nOld preprocessing failed. Cannot train old KNN model.")


# 3. Compare Metrics
print("\n" + "="*30 + " METRIC COMPARISON " + "="*30)
comparison_possible = True
if not metrics_old:
    print("Metrics for OLD preprocessing model are not available.")
    comparison_possible = False
if not metrics_imp:
    print("Metrics for IMPROVED preprocessing model are not available.")
    comparison_possible = False

if comparison_possible:
    print(f"\nModel with OLD Preprocessing (KNN):")
    print(f"  Accuracy: {metrics_old['accuracy']:.4f}")
    print(f"  Macro Avg F1-Score: {metrics_old['macro_f1']:.4f}")

    print(f"\nModel with IMPROVED Preprocessing ({best_model_info_imp['name']}):")
    print(f"  Accuracy: {metrics_imp['accuracy']:.4f}")
    print(f"  Macro Avg F1-Score: {metrics_imp['macro_f1']:.4f}")

    # Calculate and print differences
    accuracy_diff = metrics_imp['accuracy'] - metrics_old['accuracy']
    f1_diff = metrics_imp['macro_f1'] - metrics_old['macro_f1'] # Use the extracted macro_f1

    print("\n--- Difference (Improved - Old) ---")
    print(f"Accuracy Difference: {accuracy_diff:+.4f}")
    print(f"Macro Avg F1-Score Difference: {f1_diff:+.4f}")

    if accuracy_diff > 0 or f1_diff > 0:
        print("\nConclusion: The improved preprocessing pipeline resulted in better performance.")
    elif accuracy_diff < 0 or f1_diff < 0:
         print("\nConclusion: The improved preprocessing pipeline resulted in better performance. Review implementation.")
    else:
        print("\nConclusion: Both preprocessing pipelines resulted in similar performance.")

else:
    print("\nCould not perform final metric comparison due to missing results.")


print("\nScript finished.")