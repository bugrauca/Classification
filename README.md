# Experimentation with Classifiers - README.txt

## Introduction
This program is designed to conduct experiments using three different classifiers (KNeighborsClassifier, GaussianNB, MLPClassifier) on a dataset comprising image information and edge histograms. 
The goal is to train and evaluate these classifiers, saving both the confusion matrices and hyperparameters for analysis.

## Installation
1. Ensure you have Python installed on your system (version 3.6 or later).
2. Install the required libraries by running:

 `pip install numpy pandas scikit-learn`

## Execution
1. Place the program script and the dataset files (`Images.csv` and `EdgeHistogram.csv`) in the same directory.
2. Run the script:

    `python classification.py`

## Operation
1. Reading Data:
- The program reads data from `Images.csv` and `EdgeHistogram.csv`.

2. Merging Data:
- Merges the two datasets based on the 'ID' column.

3. Splitting Data:
- Splits the dataset into features (X) and labels (y).
- Further splits the data into training and testing sets.

4. Classifiers:
- Three classifiers are employed: KNeighborsClassifier, GaussianNB, and MLPClassifier.

5. Training and Evaluation:
- Each classifier is trained using the training set and evaluated on the testing set.
- Hyperparameter tuning is performed using GridSearchCV.
- Confusion matrices are computed and stored.

6. Saving Hyperparameters:
- Hyperparameters for each classifier are saved in a CSV file named `parameters{i + 1}.csv`.

7. Saving Confusion Matrices:
- Confusion matrices for each classifier are saved in CSV files named `result{i + 1}.csv`.

## Libraries and Applications
- Libraries:
- `numpy`: Numerical operations on arrays.
- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning toolkit for classification and evaluation.

- Applications:
- Utilizes scikit-learn library for classifiers, data splitting, evaluation, and hyperparameter tuning. 
- Relies on numpy and pandas for data handling and manipulation.

## Notes
- Adjust file names or paths if necessary.
- Results and hyperparameters are saved for each classifier in separate files for analysis.
