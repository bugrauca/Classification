import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

print("Reading data...")
images_df = pd.read_csv('Images.csv', sep=';', header=None, skiprows=1, names=['ID', 'Class'])
edge_histogram_df = pd.read_csv('EdgeHistogram.csv', sep=';', header=None, skiprows=1)

print("Merging data...")
merged_df = pd.merge(images_df, edge_histogram_df, left_on='ID', right_on=0)

print("Splitting data into features and labels...")
X = merged_df.iloc[:, 2:]
y = merged_df.iloc[:, 1]

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Classifiers
classifiers = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "MLPClassifier": MLPClassifier(max_iter=1000)
}

# Hyperparameters grid for each classifier
param_grids = {
    "KNeighborsClassifier": {'n_neighbors': [3, 5, 7]},
    "GaussianNB": {},
    "MLPClassifier": {'hidden_layer_sizes': [(50,), (100,), (50, 50)]}
}

# Train classifiers
results = {}
for i, (classifier_name, classifier) in enumerate(classifiers.items()):
    print(f"Training {classifier_name}...")
    if classifier_name in param_grids:
        grid_search = GridSearchCV(classifier, param_grids[classifier_name], cv=5)
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_classifier = classifier.fit(X_train, y_train)
        best_params = classifier.get_params()
    
    y_pred = best_classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    results[classifier_name] = cm

    print("Saving hyperparameters...")
    hyperparameters = {
        "classifier_name": classifier_name,
        "library": "scikit-learn",
        "test_size": 0.2
    }
    
    # Update with best parameters
    for param, value in best_params.items():
        hyperparameters[param] = value

    pd.DataFrame(hyperparameters.items(), columns=['name', 'value']).to_csv(f'parameters{i + 1}.csv', header=False, index=False)

print("Saving confusion matrices...")
for i, (classifier_name, cm) in enumerate(results.items()):
    labels = np.unique(y)
    cm_df = pd.DataFrame(cm, columns=labels, index=labels)
    cm_df.to_csv(f'result{i + 1}.csv', sep=',', index_label='')

print("Experiment completed. Results and hyperparameters saved.")