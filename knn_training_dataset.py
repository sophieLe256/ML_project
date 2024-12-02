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
def train_knn_classifier(train_data, train_labels, n_neighbors=5):
    """Train a KNN classifier."""
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_data, train_labels.values.ravel())
    return knn_model

# Step 4: Predict and Evaluate
def predict_and_evaluate(model, train_data, train_labels, test_data, dataset_num):
    """Predict test data and evaluate the model on training data."""
    # Predict on training data (for accuracy check)
    train_predictions = model.predict(train_data)
    
    # Predict on test data (final output)
    test_predictions = model.predict(test_data)
    
    # Evaluate on training data
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
all_accuracies = []

for i in datasets:
    print(f"\nProcessing Dataset {i}...")
    train_file = f'./ImputedData/TrainData{i}.xlsx'
    label_file = f'./Excel/output_TrainLabel{i}.xlsx'
    test_file = f'./ImputedData/TestData{i}.xlsx'
    output_file = f'./Output/Predictions_TestData{i}_KNN.txt'
    
    # Load the data
    train_data, train_labels, test_data = load_data(train_file, label_file, test_file)
    
    # Scale the data
    train_data_scaled, test_data_scaled = scale_data(train_data, test_data)
    
    # Train the KNN model
    knn_model = train_knn_classifier(train_data_scaled, train_labels, n_neighbors=5)
    
    # Predict and evaluate on training data and test data
    test_predictions, accuracy = predict_and_evaluate(knn_model, train_data_scaled, train_labels, test_data_scaled, i)
    
    # Save test predictions
    save_predictions(test_predictions, output_file)
    
    # Collect accuracy
    all_accuracies.append((i, accuracy * 100))

# Print Accuracy for All Datasets
print("\nAccuracy Summary for All Datasets:")
for dataset_num, acc in all_accuracies:
    print(f"Dataset {dataset_num}: Training Accuracy = {acc:.2f}%")

    import matplotlib.pyplot as plt

# Extract dataset numbers and accuracies for plotting
dataset_nums = [x[0] for x in all_accuracies]
accuracies = [x[1] for x in all_accuracies]

# Plotting the accuracy for each dataset
plt.figure(figsize=(10, 6))
plt.plot(dataset_nums, accuracies, marker='o', linestyle='-', color='b')
plt.title('KNN Accuracy for All Datasets')
plt.xlabel('Dataset Number')
plt.ylabel('Accuracy (%)')
plt.xticks(dataset_nums)  # Ensure all dataset numbers are labeled on the x-axis
plt.grid(True)
plt.savefig('./Output/KNN_Accuracy_Per_Dataset.png')
plt.show()


