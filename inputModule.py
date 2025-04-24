import pandas as pd
import numpy as np
import pickle
import json

class InputModule:
    def __init__(self):
        self.domain = None
        self.model_path = None
        self.dataset_path = None
        self.preprocessing_code = None
        self.datastet = None
        self.current_metrics_path = None
    
    def take_input(self):
        self.domain = input("Enter Domain: ")
        self.model_path = input("Enter Model Path: ")
        self.dataset_path = input("Enter Dataset Path: ")
        self.dataset = pd.read_csv(self.dataset_path)
        self.preprocessing_code = input("Enter Preprocessing Code: ")
        self.current_metrics_path =  input("Enter Current Metrics Path: ")
        
    def display_inputs(self):
        print(f"Domain: {self.domain}")
        print(f"Model Path: {self.model_path}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Preprocessing Code: {self.preprocessing_code}")

    def eda(self):
        return self.perform_eda_enhanced(self.dataset)
    
    def current_metrics(self):
        return self.perfrom_current_metrics(self.model_path, self.current_metrics_path)

    def perform_eda_enhanced(self, df: pd.DataFrame) -> dict:
        """
        ------------------------------------------------------------------------
        Function to perform an enhanced Exploratory Data Analysis (EDA) on a 
        pandas DataFrame to better understand the data for improved preprocessing.

        Parameters:
            df (pandas DataFrame): The input DataFrame for EDA.

        Returns:
            dict: A dictionary containing detailed results of the EDA, including:
                - Missing value count and percentage per column.
                - Total number of duplicate rows.
                - Column names and their data types.
                - Separated lists of numerical and categorical columns.
                - Descriptive statistics (mean, std, etc.) for numerical columns.
                - Skewness and kurtosis for numerical columns.
                - Interquartile range (IQR) for numerical columns.
                - High correlation pairs among numerical features (threshold > 0.8).


        ------------------------------------------------------------------------
        """
        eda_results = {}

        # --- Missing Values ---
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        eda_results['missing_values_count'] = missing_count.to_dict()
        eda_results['missing_values_percentage'] = missing_percentage.to_dict()

        # --- Duplicate Rows ---
        eda_results['duplicate_rows'] = int(df.duplicated().sum())

        # --- Columns and Data Types ---
        eda_results['columns'] = df.columns.tolist()
        eda_results['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()

        # --- Numerical and Categorical Columns ---
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        eda_results['numerical_columns'] = numerical_columns
        eda_results['categorical_columns'] = categorical_columns

        # --- Descriptive Statistics for Numerical Columns ---
        if numerical_columns:
            eda_results['numerical_summary'] = df[numerical_columns].describe().to_dict()
            eda_results['numerical_skewness'] = df[numerical_columns].skew().to_dict()
            eda_results['numerical_kurtosis'] = df[numerical_columns].kurtosis().to_dict()
            
            # Calculate the Interquartile Range (IQR)
            Q1 = df[numerical_columns].quantile(0.25)
            Q3 = df[numerical_columns].quantile(0.75)
            IQR = Q3 - Q1
            eda_results['numerical_IQR'] = IQR.to_dict()
        else:
            eda_results['numerical_summary'] = {}
            eda_results['numerical_skewness'] = {}
            eda_results['numerical_kurtosis'] = {}
            eda_results['numerical_IQR'] = {}

        # --- Correlation Analysis for Numerical Columns ---
        if len(numerical_columns) > 1:
            corr_matrix = df[numerical_columns].corr().abs()
            # Use a mask to extract the upper triangular matrix without the diagonal
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            upper_tri = corr_matrix.where(~mask)
            # Find pairs with correlation above 0.8 (threshold can be adjusted)
            high_corr = (upper_tri[upper_tri > 0.8]
                        .stack()
                        .reset_index())
            high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
            eda_results['high_correlation_pairs'] = high_corr.to_dict(orient='records')
        else:
            eda_results['high_correlation_pairs'] = []


        return eda_results
    
    def perfrom_current_metrics(self, model_path, current_metrics_path):
        current_metrics = {}
        current_metrics['model_name_and_hyperparameters'] = self.model_name_and_hyperparameters(model_path)
        with open(current_metrics_path, 'r') as json_file:
            evaluation_dict = json.load(json_file)
        current_metrics['evaluation_dict'] = evaluation_dict
        return current_metrics

    def model_name_and_hyperparameters(self, model_path):
        """
        Loads the pickled model and extracts its hyperparameters if available.
        """
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Inspect the type of the model
        model_type = model

        if hasattr(model, 'get_params'):
            params = model.get_params()
            return model_type, params
        else:
            print("This model does not expose hyperparameters via get_params.")
            return model_type, {}