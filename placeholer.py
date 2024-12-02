import os
import pandas as pd
# Load and inspect the uploaded datasets for placeholder values

# Files uploaded for Dataset 1, 2, 3
missing_data_1 = pd.read_excel('./Excel/output_MissingData1.xlsx', header=None)
missing_data_2 = pd.read_excel('./Excel/output_MissingData2.xlsx', header=None)
missing_data_3 = pd.read_excel('./Excel/output_MissingData3.xlsx', header=None)

# Function to verify missing value placeholders
def verify_missing_placeholders(data, dataset_name):
    unique_values = data.stack().unique()
    placeholder_count_1 = (data == 1.00000000000000e+99).sum().sum()
    placeholder_count_2 = (data == 1000000000).sum().sum()
    nan_count = data.isnull().sum().sum()
    
    print(f"--- {dataset_name} ---")
    print(f"Unique values in dataset: {unique_values[:10]} (showing up to 10 unique values)")
    print(f"Count of 1.00000000000000e+99 placeholders: {placeholder_count_1}")
    print(f"Count of 1000000000 placeholders: {placeholder_count_2}")
    print(f"Count of NaN (empty) values: {nan_count}\n")

# Verify all uploaded datasets
verify_missing_placeholders(missing_data_1, "Dataset 1")
verify_missing_placeholders(missing_data_2, "Dataset 2")
verify_missing_placeholders(missing_data_3, "Dataset 3")
