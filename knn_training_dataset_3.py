import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Step 3: Apply PCA
def apply_pca_and_filter(train_data, test_data, n_components):
    """Apply PCA to reduce dimensions and keep top components."""
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_data)
    test_pca = pca.transform(test_data)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained Variance Ratio with {n_components} components: {explained_variance:.2f}%")
    return train_pca, test_pca

# Step 4: Train and Evaluate KNN
def train_and_evaluate_knn(train_data, train_labels, test_data, k):
    """Train KNN and evaluate on training data."""
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(train_data, train_labels.values.ravel())
    
    # Training accuracy
    train_predictions = knn.predict(train_data)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f"K={k}, Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Test predictions
    test_predictions = knn.predict(test_data)
    return train_accuracy, test_predictions

# Step 5: Save Predictions
def save_predictions(predictions, output_file):
    """Save predictions to a text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Step 6: Plot Accuracy vs. K
def plot_accuracy(k_values, accuracy_list):
    """Plot accuracy for different K values."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_list, marker='o')
    plt.title('KNN Accuracy for Different K Values (Dataset 3)')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('./Output/KNN_Accuracy_Dataset3.png')
    plt.show()

# Main Workflow for Dataset 3
print("\nProcessing Dataset 3 with PCA and KNN...")
train_file = './ImputedData/Imputed_TrainData3.xlsx'
label_file = './Excel/output_TrainLabel3.xlsx'
test_file = './ImputedData/Imputed_TestData3.xlsx'
output_file = './Output/Predictions_TestData3_PCA_KNN.txt'

# Load and scale the data
train_data, train_labels, test_data = load_data(train_file, label_file, test_file)
train_data_scaled, test_data_scaled = scale_data(train_data, test_data)

# Apply PCA to reduce features to the first 9 components
train_pca, test_pca = apply_pca_and_filter(train_data_scaled, test_data_scaled, n_components=9)

# Test K values from 2 to 13
accuracy_list = []
k_values = range(2, 14)

for k in k_values:
    train_accuracy, test_predictions = train_and_evaluate_knn(train_pca, train_labels, test_pca, k)
    accuracy_list.append(train_accuracy)

# Plot accuracy for different K values
plot_accuracy(k_values, accuracy_list)

# Train final model with the best K
best_k = k_values[np.argmax(accuracy_list)]
final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
final_model.fit(train_pca, train_labels.values.ravel())
final_test_predictions = final_model.predict(test_pca)
save_predictions(final_test_predictions, output_file)
