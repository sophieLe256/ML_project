import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def impute_missing_values(data, method='mean', n_neighbors=5):
    """Impute missing values with the specified method."""
    # Ensure missing placeholders are replaced with np.nan
    data.replace(1.00000000000000e+99, np.nan, inplace=True)
    
    # Imputation method
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        raise ValueError("Invalid imputation method.")
    
    # Fit and transform data
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return imputed_data

# Example usage for Dataset 1
data1 = pd.read_excel('./Excel/output_MissingData1.xlsx', header=None)
imputed_data1 = impute_missing_values(data1, method='mean')
imputed_data1.to_excel('./ImputedData/Imputed_MissingData1.xlsx', index=False, header=False)
