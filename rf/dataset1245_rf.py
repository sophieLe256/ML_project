import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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
def train_rf_classifier(train_data, train_labels, n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42):
    """Train a Random Forest classifier with regularization."""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    rf_model.fit(train_data, train_labels.values.ravel())
    return rf_model

# Step 4: Evaluate with Cross-Validation
def cross_validate_model(model, train_data, train_labels, cv=5):
    """Perform cross-validation and return mean accuracy."""
    scores = cross_val_score(model, train_data, train_labels.values.ravel(), cv=cv)
    print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")
    return scores.mean()

# Step 5: Predict and Evaluate
def predict_and_evaluate(model, train_data, train_labels, test_data, dataset_num):
    """Predict test data and evaluate the model on training data."""
    # Predict on training data
    train_predictions = model.predict(train_data)
    
    # Predict on test data
    test_predictions = model.predict(test_data)
    
    # Evaluate on training data
    accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Dataset {dataset_num} - Training Accuracy: {accuracy * 100:.2f}%")
    print(f"Dataset {dataset_num} - Classification Report on Training Data:\n",
          classification_report(train_labels, train_predictions))
    print(f"Dataset {dataset_num} - Confusion Matrix on Training Data:\n",
          confusion_matrix(train_labels, train_predictions))
    
    return test_predictions, accuracy

# Step 6: Save Predictions
def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Main Workflow for Dataset 1,2,4,5
datasets = [
    {"train_file": "./ImputedData/TrainData1.xlsx", "label_file": "./Excel/output_TrainLabel1.xlsx", "test_file": "./ImputedData/TestData1.xlsx", 
     "output_file": "./Output/LeClassification1_rf.txt", "dataset_num": 1},
    {"train_file": "./ImputedData/TrainData2.xlsx", "label_file": "./Excel/output_TrainLabel2.xlsx", "test_file": "./ImputedData/TestData2.xlsx", 
     "output_file": "./Output/LeClassification2_rf.txt", "dataset_num": 2},
    {"train_file": "./ImputedData/TrainData4.xlsx", "label_file": "./Excel/output_TrainLabel4.xlsx", "test_file": "./ImputedData/TestData4.xlsx", 
    "output_file": "./Output/LeClassification4_rf.txt", "dataset_num": 4},
    {"train_file": "./ImputedData/TrainData5.xlsx", "label_file": "./Excel/output_TrainLabel5.xlsx", "test_file": "./ImputedData/TestData5.xlsx", 
    "output_file": "./Output/LeClassification5_rf.txt", "dataset_num": 5}
]

for dataset in datasets:
    print(f"\nProcessing Dataset {dataset['dataset_num']}...")
    train_file = dataset["train_file"]
    label_file = dataset["label_file"]
    test_file = dataset["test_file"]
    output_file = dataset["output_file"]
    dataset_num = dataset["dataset_num"]

    train_data, train_labels, test_data = load_data(train_file, label_file, test_file)

    train_data_scaled, test_data_scaled = scale_data(train_data, test_data)

    X_train, X_val, y_train, y_val = train_test_split(train_data_scaled, train_labels, test_size=0.2, random_state=42)

    rf_model = train_rf_classifier(X_train, y_train)

    cross_validate_model(rf_model, train_data_scaled, train_labels)

    # Predict and evaluate on training and test data
    test_predictions, accuracy = predict_and_evaluate(rf_model, train_data_scaled, train_labels, test_data_scaled, dataset_num)

    # Save test predictions
    save_predictions(test_predictions, output_file)
