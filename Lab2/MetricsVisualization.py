# import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import sklearn
import sklearn.preprocessing
import sklearn.metrics
import sklearn.svm
import sklearn.tree
import sklearn.neighbors

# Import Data and Process
X = pd.read_csv("Iris.csv")
y = X["Species"]
X.drop("Id", axis=1, inplace=True)
X.drop("Species", axis=1, inplace=True)

# 80/20 train/test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20)

# standardize data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create classifier
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)

# train classifier
knn.fit(X_train, y_train)

# Confusion Matrix
sklearn.metrics.ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.savefig('CM.pdf')
