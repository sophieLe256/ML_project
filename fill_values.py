import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Step 1: Load Data
def load_data(file_path):
    return pd.read_excel(file_path, header=None)

# Step 2: Determine Imputation Strategy Based on Missing Data Percentage
def determine_strategy(data, missing_threshold=0.2):
    # Replace missing value placeholder (1.00000000000000e+99) with NaN
    data.replace(1000000000, np.nan, inplace=True)
    data.replace(1.00000000000000e+99, np.nan, inplace=True)

    # Calculate percentage of missing data for each column
    missing_percentage = data.isnull().mean().mean()
    print('Missing percentage:', missing_percentage)
    
    # If the percentage of missing data exceeds the threshold, use KNN
    if missing_percentage > missing_threshold:
        return 'knn'
    
    # Otherwise, use mean/median based on skewness
    skewness = data.skew().mean()
    return 'median' if abs(skewness) > 0.5 else 'mean'

# Step 3: Impute Missing Values
def impute_missing_values(data, method='mean'):
    print('method used:', method)

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
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    return imputed_data

# Step 4: Save Imputed Data
def save_imputed_data(imputed_data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imputed_data.to_excel(output_file, index=False, header=False)
    print(f"Imputed data saved to {output_file}")

# Main Workflow
def main_workflow(datasets, missing_threshold=0.2):
    # Impute data for traindata
    for i in datasets:
        input_file = f'./Excel/output_TrainData{i}.xlsx'
        output_file = f'./ImputedData/TrainData{i}.xlsx'
        
        data = load_data(input_file)
        
        method = determine_strategy(data, missing_threshold)
        imputed_data = impute_missing_values(data, method=method)
        
        save_imputed_data(imputed_data, output_file)
    
    # Impute data for traindata
    for i in datasets:
        input_file = f'./Excel/output_TestData{i}.xlsx'
        output_file = f'./ImputedData/TestData{i}.xlsx'
        
        data = load_data(input_file)
        
        method = determine_strategy(data, missing_threshold)
        imputed_data = impute_missing_values(data, method=method)
        
        save_imputed_data(imputed_data, output_file)

    # Impute data for missing values
    for i in range(1, 4):
        input_file = f'./Excel/output_MissingData{i}.xlsx'
        output_file = f'./MissingValues/MissingData{i}.xlsx'
        
        data = load_data(input_file)
        
        method = determine_strategy(data, missing_threshold)
        imputed_data = impute_missing_values(data, method=method)
        
        save_imputed_data(imputed_data, output_file)

datasets = [1, 2, 3, 4, 5] 
missing_threshold = 0.2

main_workflow(datasets, missing_threshold)
