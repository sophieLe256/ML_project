import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Data
def load_data(train_file, label_file, test_file):
    """Load training data, labels, and test data."""
    train_data = pd.read_excel(train_file, header=None)
    train_labels = pd.read_excel(label_file, header=None)
    test_data = pd.read_excel(test_file, header=None)
    return train_data, train_labels, test_data

# Step 2: Scale Data
def scale_data(train_data, test_data):
    """Scale data using StandardScaler."""
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled

# Step 3: Train Random Forest Classifier
def train_rf_classifier(train_data, train_labels, n_estimators=100, max_depth=None):
    """Train a Random Forest classifier."""
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(train_data, train_labels.values.ravel())
    return rf_model

# Step 4: Predict and Evaluate
def predict_and_evaluate_rf(model, train_data, train_labels, test_data, dataset_num):
    """Predict test data and evaluate the model on training data."""
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)
    
    accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Dataset {dataset_num} - Training Accuracy: {accuracy * 100:.2f}%")
    print(f"Dataset {dataset_num} - Classification Report on Training Data:\n", 
          classification_report(train_labels, train_predictions))
    print(f"Dataset {dataset_num} - Confusion Matrix on Training Data:\n", 
          confusion_matrix(train_labels, train_predictions))
    
    return test_predictions, accuracy

# Step 5: Save Predictions
def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Main Workflow
datasets = [1, 2, 3, 4, 5]  # Specify dataset numbers
rf_accuracies = []

for i in datasets:
    print(f"\nProcessing Dataset {i} with Random Forest...")
    train_file = f'./ImputedData/TrainData{i}.xlsx'
    label_file = f'./Excel/output_TrainLabel{i}.xlsx'
    test_file = f'./ImputedData/TestData{i}.xlsx'
    output_file = f'./Output/Predictions_TestData{i}_RF.txt'
    
    train_data, train_labels, test_data = load_data(train_file, label_file, test_file)
    train_data_scaled, test_data_scaled = scale_data(train_data, test_data)
    
    rf_model = train_rf_classifier(train_data_scaled, train_labels, n_estimators=100, max_depth=10)
    test_predictions, accuracy = predict_and_evaluate_rf(rf_model, train_data_scaled, train_labels, test_data_scaled, i)
    
    save_predictions(test_predictions, output_file)
    rf_accuracies.append((i, accuracy * 100))

# Print Accuracy Summary
print("\nRandom Forest Accuracy Summary for All Datasets:")
for dataset_num, acc in rf_accuracies:
    print(f"Dataset {dataset_num}: Training Accuracy = {acc:.2f}%")
