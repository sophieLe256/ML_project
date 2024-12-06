import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Data
def load_data():
    train_data = pd.read_excel('./ImputedData/TrainData5.xlsx', header=None)
    train_labels = pd.read_excel('./Excel/output_TrainLabel5.xlsx', header=None)
    test_data = pd.read_excel('./ImputedData/TestData5.xlsx', header=None)
    return train_data, train_labels, test_data

# Step 2: Handle Missing Values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(data))

# Step 3: Scale Data
def scale_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled

# Step 4: Train and Evaluate KNN
def train_and_evaluate_knn(train_data, train_labels, test_data, n_neighbors=5):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_data, train_labels.values.ravel())
    
    train_predictions = knn_model.predict(train_data)
    test_predictions = knn_model.predict(test_data)
    
    accuracy = accuracy_score(train_labels, train_predictions)
    print("Training Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(train_labels, train_predictions))
    print("Confusion Matrix:\n", confusion_matrix(train_labels, train_predictions))
    
    return test_predictions

# Step 5: Save Predictions
def save_predictions(predictions):
    os.makedirs('./Output', exist_ok=True)
    pd.DataFrame(predictions).to_csv('./Output/Predictions_TestData5_KNN.txt', index=False, header=False)
    print("Predictions saved to './Output/LeClassification5_knn.txt'")

# Main Workflow
if __name__ == "__main__":
    train_data, train_labels, test_data = load_data()
    
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    train_data_scaled, test_data_scaled = scale_data(train_data, test_data)
    
    test_predictions = train_and_evaluate_knn(train_data_scaled, train_labels, test_data_scaled, n_neighbors=5)
    save_predictions(test_predictions)
