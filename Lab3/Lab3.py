#############################################################
#
# Name: 
# 
# Collaborators:
#
# ############################################################

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, model_selection, preprocessing, svm, metrics, decomposition, neighbors, linear_model
# ADDITIONAL IMPORTS AS NEEDED


def fit_predict():
    """Complete this function as described in Lab3.pdf"""

    X_train = pd.read_csv("Lab3_X_train.csv")
    y_train = pd.read_csv("Lab3_y_train.csv").squeeze("columns")
    X_test = pd.read_csv("Lab3_X_test.csv")
    
    
    group_means = X_train.groupby("Location").mean()
    X_train = X_train.groupby("Location").apply(lambda x: x.fillna(group_means))
    X_test = X_test.groupby("Location").apply(lambda x: x.fillna(group_means))
   
    #sns.pairplot(X_train, hue="Location")
    #temp = X_train.join(y_train)
    #sns.pairplot(temp, hue="RainTomorrow")
    
    #X_test = X_test.groupby(['Location']).fillna(Gr_means)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    means = np.mean(X_train)

    X_train = X_train.fillna(means)
    X_test = X_test.fillna(means)

    ss = preprocessing.StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)


    #reg = linear_model.LogisticRegression()
    #reg = linear_model.LogisticRegression(fit_intercept=False)
    #reg = linear_model.LogisticRegression(fit_intercept=False, n_jobs = 2)
    #reg = linear_model.LogisticRegression(n_jobs = 2)
    #reg = linear_model.LogisticRegression(n_jobs = 10)
    #reg = linear_model.LogisticRegression(n_jobs = 50)
    #reg = linear_model.LogisticRegression(n_jobs = 100)
    #reg = linear_model.LogisticRegression(n_jobs = 250)
    reg = linear_model.LogisticRegression()
    reg.fit(X_train, y_train)

    #svc = svm.SVC()
    #svc.fit(X_train, y_train)
    
    # Predict labels for the test data
    y_pred = reg.predict(X_test)
    #y_pred = svc.predict(X_test)
   
    
    # return predicted test labels
    return y_pred


def main():
    """This function is for your own testing. It will not be called by the leaderboard."""
    #pass
    fit_predict()

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
