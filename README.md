# Classification Using KNN and Random Forest
[![GitHub stars](https://img.shields.io/badge/Stars-0-yellow.svg?style=flat-square)](https://github.com/username/repository/stargazers)
[![Maintainability](https://img.shields.io/badge/Maintainability-100%25-brightgreen.svg?style=flat-square)](https://codeclimate.com/github/username/repository)

## :computer: Description
This project implements classification for five datasets using two algorithms:
K-Nearest Neighbors (KNN)
Random Forest (RF)

The objective is to compare the performance of both algorithms and select the best model for each dataset. The target accuracy thresholds are:

- Dataset 1, 2, 4: Achieve > 90% accuracy
* Dataset 3, 6: Achieve > 60% accuracy
### Selected Models for Each Dataset
- Dataset 1, 2, 4, 5: Best accuracy achieved with KNN.
* Dataset 3: Best accuracy achieved with Random Forest (RF).

### Libraries Used
The following libraries are required to run this project:

- Pandas: For data manipulation and reading .xlsx files.
* NumPy: For numerical computations.
+ Scikit-learn: For implementing machine learning models (KNN and RF), scaling data, and evaluating performance.
- OpenPyXL: For reading and writing Excel files.

## :star2: How to Clone and Run the ML Project

### 1. Clone the Repository
1. To get started with the project, clone the repository to your local machine:

```bash
git clone https://github.com/sophieLe256/ML_project.git
```
2. Navigate into the project directory using the terminal or command prompt:
```
cd ML_project
```

### 2. Set Up the Environment
1. Ensure you have Python 3.8 or higher installed on your system. You can check by running:
```bash
python --version
```
2. Install the required libraries
```bash
pip install pandas numpy scikit-learn openpyxl
```

### 3. Convert TXT Files to Excel
> [!IMPORTANT]
If your dataset files are in `txt` format, you need to convert them to Excel format `(.xlsx)` before running the algorithms. Follow these steps:

1. Open the `convert_to_excel.py` script included in the repository.
2. Update the input file paths (txt files) and output file paths (Excel files) in the script.
3. Run the script to generate Excel files:
```bash
python convert_to_excel.py
```
The converted files will be saved in the appropriate directories `Excel/`.

### 4: Prepare the Data
Ensure the following directory structure:

Training and test data (`TrainData*.xlsx`, `TestData*.xlsx`) in the `ImputedData/` folder.
Training labels (`output_TrainLabel*.xlsx`) in the `Excel/` folder.

### 5: Run the Classification Script
Run the main Python script to process for each datasets:
Example: Dataset 1
```bash
python3 ./knn/dataset1_knn.py
```

### 6: View the Result
1. Predicted test labels for each dataset will be saved in the `Output/` folder:

- LeClassification1_knn.txt
- LeClassification1_rf.txt
* LeClassification2_knn.txt
* LeClassification2_rf.txt
+ LeClassification3_knn.txt
+ LeClassification3_rf.txt
- LeClassification4_knn.txt
- LeClassification4_rf.txt
* LeClassification5_knn.txt
* LeClassification5_rf.txt
  
2. The console output will display:
- Accuracy for KNN and RF models for each dataset.
* The selected best model for each dataset.
+ Training accuracy, precision, recall, and confusion matrix.

