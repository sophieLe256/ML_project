import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Step 3: Train KNN Classifier
def train_knn_classifier(train_data, train_labels, n_neighbors=1):
    """Train a KNN classifier with k=5."""
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_data, train_labels.values.ravel())
    return knn_model

# Step 4: Predict and Evaluate
def predict_and_evaluate(model, train_data, train_labels, test_data):
    """Predict test data and evaluate the model on training data."""
    # Predict on training data (for accuracy check)
    train_predictions = model.predict(train_data)
    
    # Predict on test data (final output)
    test_predictions = model.predict(test_data)
    
    # Evaluate on training data
    accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Dataset 1 - Training Accuracy: {accuracy * 100:.2f}%")
    print("Dataset 1 - Classification Report on Training Data:\n", 
          classification_report(train_labels, train_predictions))
    print("Dataset 1 - Confusion Matrix on Training Data:\n", 
          confusion_matrix(train_labels, train_predictions))
    
    return test_predictions, accuracy

# Step 5: Save Predictions
def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Main Workflow for Dataset 2
print("\nProcessing Dataset 1...")
train_file = './ImputedData/TrainData1.xlsx'
label_file = './Excel/output_TrainLabel1.xlsx'
test_file = './ImputedData/TestData1.xlsx'
output_file = './Output/LeClassification1_knn.txt'

# Load the data
train_data, train_labels, test_data = load_data(train_file, label_file, test_file)

# Scale the data
train_data_scaled, test_data_scaled = scale_data(train_data, test_data)

# Train the KNN model with k=3
knn_model = train_knn_classifier(train_data_scaled, train_labels, n_neighbors=3)

# Predict and evaluate on training data and test data
test_predictions, accuracy = predict_and_evaluate(knn_model, train_data_scaled, train_labels, test_data_scaled)

# Save test predictions
save_predictions(test_predictions, output_file)

# Accuracy Summary
print(f"\nDataset 1: Training Accuracy = {accuracy * 100:.2f}%")

