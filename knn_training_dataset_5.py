import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
def train_knn_classifier(train_data, train_labels, n_neighbors):
    """Train a KNN classifier."""
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_data, train_labels.values.ravel())
    return knn_model

# Step 4: Evaluate Model
def evaluate_model(model, train_data, train_labels):
    """Evaluate the model on the training data."""
    train_predictions = model.predict(train_data)
    accuracy = accuracy_score(train_labels, train_predictions)
    return accuracy

# Step 5: Save Predictions
def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Step 6: Plot Accuracy vs. K
def plot_accuracy(accuracy_list, k_values):
    """Plot accuracy for each K value."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_list, marker='o')
    plt.title('KNN Accuracy for Different K Values (Dataset 5)')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('./Output/KNN_Accuracy_Dataset5.png')
    plt.show()

# Main Workflow for Dataset 5
print("\nProcessing Dataset 5...")
train_file = './ImputedData/Imputed_TrainData5.xlsx'
label_file = './Excel/output_TrainLabel5.xlsx'
test_file = './ImputedData/Imputed_TestData5.xlsx'
output_file = './Output/Predictions_TestData5_KNN.txt'

# Load and scale the data
train_data, train_labels, test_data = load_data(train_file, label_file, test_file)
train_data_scaled, test_data_scaled = scale_data(train_data, test_data)

# Test K values from 1 to 20
accuracy_list = []
k_values = range(2, 13)

for k in k_values:
    knn_model = train_knn_classifier(train_data_scaled, train_labels, n_neighbors=k)
    accuracy = evaluate_model(knn_model, train_data_scaled, train_labels)
    accuracy_list.append(accuracy)
    print(f"K={k}, Accuracy={accuracy * 100:.2f}%")

# Plot accuracy for all K values
plot_accuracy(accuracy_list, k_values)

# Train final model with the best K
best_k = k_values[np.argmax(accuracy_list)]
final_model = train_knn_classifier(train_data_scaled, train_labels, n_neighbors=best_k)
test_predictions = final_model.predict(test_data_scaled)
save_predictions(test_predictions, output_file)
