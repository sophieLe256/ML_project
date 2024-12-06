import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Step 1: Load Data
def load_data():
    train_data = pd.read_excel('./ImputedData/TrainData3.xlsx', header=None)
    train_labels = pd.read_excel('./Excel/output_TrainLabel3.xlsx', header=None)
    test_data = pd.read_excel('./ImputedData/TestData3.xlsx', header=None)
    return train_data, train_labels, test_data

# Step 2: Handle Missing Values
def handle_missing_values(data):
    imputer = KNNImputer(n_neighbors=5)
    return pd.DataFrame(imputer.fit_transform(data))

# Step 3: Scale Data
def scale_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled

# Step 4: Optimize Random Forest
def optimize_random_forest(train_data, train_labels):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(train_data, train_labels.values.ravel())
    return grid_search.best_params_

# Step 5: Train and Evaluate Random Forest
def train_and_evaluate_rf(train_data, train_labels, test_data, best_params):
    rf_model = RandomForestClassifier(random_state=42, **best_params)
    rf_model.fit(train_data, train_labels.values.ravel())
    
    train_predictions = rf_model.predict(train_data)
    test_predictions = rf_model.predict(test_data)
    
    accuracy = accuracy_score(train_labels, train_predictions)
    print("Training Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(train_labels, train_predictions))
    print("Confusion Matrix:\n", confusion_matrix(train_labels, train_predictions))
    
    return test_predictions

# Step 6: Save Predictions
def save_predictions(predictions):
    os.makedirs('./Output', exist_ok=True)
    pd.DataFrame(predictions).to_csv('./Output/LeClassification3.txt', index=False, header=False)
    print("Predictions saved to './Output/LeClassification3_rf.txt'")

# Main Workflow
if __name__ == "__main__":
    train_data, train_labels, test_data = load_data()
    
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    train_data_scaled, test_data_scaled = scale_data(train_data, test_data)
    
    best_params = optimize_random_forest(train_data_scaled, train_labels)
    print("Best Random Forest Parameters:", best_params)
    
    test_predictions = train_and_evaluate_rf(train_data_scaled, train_labels, test_data_scaled, best_params)
    save_predictions(test_predictions)
