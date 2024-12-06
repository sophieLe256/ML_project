import os
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore', category=UserWarning)

# Step 1: Load Data
def load_data(file_path):
    """Load data from an Excel file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_excel(file_path, header=None)

# Step 2: Determine Imputation Strategy
def determine_strategy(data, missing_threshold=0.2):
    """Determine the imputation strategy based on missing data percentage."""
    data.replace([1.00000000000000e+99, 1000000000], np.nan, inplace=True)
    missing_percentage = data.isnull().mean().mean()
    print(f'Missing Percentage: {missing_percentage:.2f}')
    if missing_percentage > missing_threshold:
        return 'knn'
    skewness = data.skew().mean()
    return 'median' if abs(skewness) > 0.5 else 'mean'

# Step 3: Impute Missing Values
def impute_missing_values(data, method='mean'):
    """Impute missing values using the specified method."""
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', or 'knn'.")
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Step 4: Scale Data
def scale_data(train_data, test_data):
    """Scale data using StandardScaler."""
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled

# Step 5: Normalize Data
def normalize_data(data):
    """Normalize data using MaxAbsScaler."""
    scaler = MaxAbsScaler()
    return scaler.fit_transform(data)

# Step 6: Check for Invalid Values
def check_for_invalid_values(data, label="Dataset"):
    """Check for invalid values (NaN, infinity, overly large values)."""
    if np.any(np.isinf(data)):
        print(f"{label} contains infinity.")
    if np.any(np.isnan(data)):
        print(f"{label} contains NaN values.")
    if np.any(data > np.finfo(np.float32).max):
        print(f"{label} contains values too large for float32.")
    if np.any(data < np.finfo(np.float32).min):
        print(f"{label} contains values too small for float32.")

# Step 7: Optimize Random Forest
def optimize_random_forest(train_data, train_labels):
    """Optimize Random Forest hyperparameters."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(train_data, train_labels.values.ravel())
    return grid_search.best_params_, grid_search.best_score_

# Step 8: Train Random Forest Classifier
def train_rf_classifier(train_data, train_labels, **rf_params):
    """Train a Random Forest classifier with specified parameters."""
    rf_model = RandomForestClassifier(random_state=42, **rf_params)
    rf_model.fit(train_data, train_labels.values.ravel())
    return rf_model

# Step 9: Predict and Evaluate Random Forest
def predict_and_evaluate_rf(model, train_data, train_labels, test_data, dataset_num):
    """Predict test data and evaluate the model on training data."""
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)
    
    # Evaluate the model on training data
    accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Dataset {dataset_num} - Random Forest Training Accuracy: {accuracy * 100:.2f}%")
    print(f"Dataset {dataset_num} - Classification Report on Training Data:\n", 
          classification_report(train_labels, train_predictions))
    print(f"Dataset {dataset_num} - Confusion Matrix on Training Data:\n", 
          confusion_matrix(train_labels, train_predictions))
    
    return test_predictions, accuracy

# Step 10: Save Predictions
def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Main Workflow
def main_workflow(datasets, missing_threshold=0.2):
    rf_results = []

    for i in datasets:
        print(f"\nProcessing Dataset {i}...")
        
        # File paths
        train_file = os.path.join('Excel', f'output_TrainData{i}.xlsx')
        label_file = os.path.join('Excel', f'output_TrainLabel{i}.xlsx')
        test_file = os.path.join('Excel', f'output_TestData{i}.xlsx')
        output_file_rf = os.path.join('Output', f'Predictions_TestData{i}_RF.txt')

        # Load data
        train_data = load_data(train_file)
        train_labels = load_data(label_file)
        test_data = load_data(test_file)

        # Impute missing values
        method = determine_strategy(train_data, missing_threshold)
        train_data = impute_missing_values(train_data, method)
        test_data = impute_missing_values(test_data, method)

        # Scale and normalize data
        train_data_scaled, test_data_scaled = scale_data(train_data, test_data)
        train_data_scaled = normalize_data(train_data_scaled)
        test_data_scaled = normalize_data(test_data_scaled)

        # Check for invalid values
        check_for_invalid_values(train_data_scaled, label="Train Data Scaled")
        check_for_invalid_values(test_data_scaled, label="Test Data Scaled")

        # Optimize Random Forest
        best_rf_params, rf_score = optimize_random_forest(train_data_scaled, train_labels)
        rf_results.append((i, best_rf_params, rf_score))

        # Train and Evaluate Random Forest with best parameters
        rf_model = train_rf_classifier(train_data_scaled, train_labels, **best_rf_params)
        test_predictions_rf, rf_accuracy = predict_and_evaluate_rf(rf_model, train_data_scaled, train_labels, test_data_scaled, i)

        # Save Random Forest Predictions
        save_predictions(test_predictions_rf, output_file_rf)

    # Output Results
    print("\nRandom Forest Optimization Results:")
    for dataset_num, params, score in rf_results:
        print(f"Dataset {dataset_num}: Best Parameters = {params}, Cross-Validated Accuracy = {score:.2f}")

# Datasets and Threshold
datasets = [1, 2, 3, 4, 5]
missing_threshold = 0.2

# Run the Main Workflow
main_workflow(datasets, missing_threshold)
