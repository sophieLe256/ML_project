import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Step 1: Load Data
def load_data(file_path):
    """Load data from Excel file."""
    return pd.read_excel(file_path, header=None)

def determine_strategy(data):
    """Determine imputation strategy based on skewness of the data."""
    skewness = data.skew().mean()
    return 'median' if abs(skewness) > 0.5 else 'mean'

# Step 2: Impute Missing Values
def impute_missing_values(data, method='mean'):
    """Impute missing values using specified strategy."""
    # Replace missing value placeholder (1.00000000000000e+99) with NaN
    data.replace(1.00000000000000e+99, np.nan, inplace=True)
    data.replace(1000000000, np.nan, inplace=True)

    # Choose imputation strategy
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', or 'knn'.")
    
    # Perform imputation
    try:
        imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    except ValueError as e:
        print(f"Imputation failed for method '{method}': {e}")
        raise
    
    # Verify if any missing values remain
    missing_count = imputed_data.isnull().sum().sum()
    print(f"Remaining missing values after {method} imputation: {missing_count}")
    
    return imputed_data

# Step 4: Save Imputed Data
def save_imputed_data(imputed_data, output_file):
    """Save imputed data to Excel file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imputed_data.to_excel(output_file, index=False, header=False)
    print(f"Imputed data saved to {output_file}")

# Main Workflow
def main_workflow(datasets, knn_dataset):
    for i in datasets:
        input_file = f'./Excel/output_TestData{i}.xlsx'
        output_file = f'./ImputedData/Imputed_TestData{i}.xlsx'
        
        data = load_data(input_file)
        
        # Use KNN for dataset 5, otherwise use determined strategy
        if i == knn_dataset:
            method = 'knn'
        else:
            method = determine_strategy(data)
        
        try:
            imputed_data = impute_missing_values(data, method=method)
        except ValueError:
            print(f"Retrying Dataset {i} with 'mean' as fallback.")
            imputed_data = impute_missing_values(data, method='mean')
        
        save_imputed_data(imputed_data, output_file)

# Define datasets and KNN-specific dataset
datasets = [1, 2, 3, 4, 5]  # All datasets
knn_dataset = 5  # Only Dataset 5 uses KNN

main_workflow(datasets, knn_dataset)
