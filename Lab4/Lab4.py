################################################
#
# Name: Seth Rosen
#
# Partner (if applicable):
#
################################################

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets, model_selection, preprocessing, svm, metrics, linear_model, ensemble
# ADDITIONAL IMPORTS AS NEEDED


def fit_predict():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")
    X_test = pd.read_csv("movies_X_test.csv")

    # YOUR CODE HERE
    reg = ensemble.RandomForestRegressor(n_estimators = 64, max_depth = 18, n_jobs = -1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # return predicted test labels 
    return y_pred


def optimize():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")
    X_test = pd.read_csv("movies_X_test.csv")
    
    k = 10
    # YOUR CODE HERE
    params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':['scale','auto'], 'shrinking':[True, False]}
    params = {'kernel':['rbf'], 'gamma':['auto']}

    gs_for = model_selection.GridSearchCV(estimator = svm.SVR(),
                                              cv = k,
                                              param_grid = params,
                                              return_train_score = True,
                                              scoring = 'neg_mean_absolute_error',
                                              n_jobs = -1,
                                              refit = True)
    gs_for.fit(X_train, y_train)
    gs_for_res = pd.DataFrame(gs_for.cv_results_).sort_values(by=['mean_test_score'], ascending = False)
    #print(gs_for.best_estimator_)



def forest():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")
    
    k = 10
    # Hyperparameters to tune:
    params = {'n_estimators': [64], 'max_depth' :[18], 'n_jobs':[-1]}

    # Initialize GridSearchCV object with decision tree classifier and hyperparameters
    gs_for = model_selection.GridSearchCV(estimator = ensemble.RandomForestRegressor(),
                                              param_grid = params,
                                              cv = k,
                                              return_train_score = True,
                                              scoring = 'neg_mean_absolute_error',
                                              n_jobs = -1,
                                              refit = True)

    # Train and cross-validate, print results
    gs_for.fit(X_train, y_train)
    gs_for_res = pd.DataFrame(gs_for.cv_results_).sort_values(by=['mean_test_score'], ascending = False)
    #print(gs_for.best_estimator_)
    


def evaluate():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")

    # YOUR CODE HERE
    

    mean_train = []
    mean_tests = []

    for x in range(0,5):
        mean_train.append(np.mean(train_scores[x]))
        mean_tests.append(np.mean(test_scores[x]))

    sns.lineplot(x = mean_train, y=train_sizes, label="Train score")
    sns.lineplot(x = mean_train, y=train_sizes, label="Test score")
    plt.savefig('learning_curve.pdf')

    #get best features with optimal forest regressor
    gs_for = ensemble.RandomForestRegressor(n_estimators=64, max_depth=18, n_jobs=-1)

    # Train and cross-validate, print results
    gs_for.fit(X_train, y_train)

    gs_for_res = pd.DataFrame(data=gs_for.feature_importances_, columns=['feat_imp'])
    names = pd.DataFrame(data=gs_for.feature_names_in_, columns=['name'])
    feats = names.join(gs_for_res)
    feats = feats.sort_values(by='feat_imp', ascending=False)
    print(feats.head(5))



def main():
    """Use this function for your own testing. It will not be called by the leaderboard"""
    pass


if __name__ == "__main__":
    main()
