import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(train_file, label_file, test_file):
    """Load training data, labels, and test data."""
    train_data = pd.read_excel(train_file, header=None)
    train_labels = pd.read_excel(label_file, header=None)
    test_data = pd.read_excel(test_file, header=None)
    return train_data, train_labels, test_data

def scale_data(train_data, test_data):
    """Scale data using StandardScaler."""
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled

def train_knn_classifier(train_data, train_labels, n_neighbors):
    """Train a KNN classifier with k=4."""
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_data, train_labels.values.ravel())
    return knn_model

def predict_and_evaluate(model, train_data, train_labels, test_data):
    """Predict test data and evaluate the model on training data."""
    train_predictions = model.predict(train_data)
    
    test_predictions = model.predict(test_data)
    
    accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Dataset 5 - Training Accuracy: {accuracy * 100:.2f}%")
    print("Dataset 5 - Classification Report on Training Data:\n", 
          classification_report(train_labels, train_predictions))
    print("Dataset 5 - Confusion Matrix on Training Data:\n", 
          confusion_matrix(train_labels, train_predictions))
    
    return test_predictions, accuracy

def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

print("\nProcessing Dataset 5...")
train_file = './ImputedData/TrainData5.xlsx'
label_file = './Excel/output_TrainLabel5.xlsx'
test_file = './ImputedData/TestData5.xlsx'
output_file = './Output/LeClassification4_knn.txt'

train_data, train_labels, test_data = load_data(train_file, label_file, test_file)

train_data_scaled, test_data_scaled = scale_data(train_data, test_data)

knn_model = train_knn_classifier(train_data_scaled, train_labels, n_neighbors=4)

test_predictions, accuracy = predict_and_evaluate(knn_model, train_data_scaled, train_labels, test_data_scaled)

save_predictions(test_predictions, output_file)

print(f"\nDataset 5: Training Accuracy = {accuracy * 100:.2f}%")

