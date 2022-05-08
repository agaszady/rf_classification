# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

#import file with features
features = pd.read_csv(r"cechy.csv", decimal=",", sep=";")
X = np.array(features.drop(features.columns[[0, 1]], axis=1))
y = np.array(features["-- Rodzaj --"])
#wrapper method (evaluates on a specific machine learning algorithm to find optimal features)
def classification_wrapper():
    print('Classification wrapper 1 method')
    clf = RandomForestClassifier(n_estimators=100)

    #initializing SelectFromModel, threshold is optional
    #threshold=0.05 - select features with importance above 0.05
    m_important = SelectFromModel(clf)

    #train test split (X - dataframe with features, y - classes)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)

    #returns transformed array with selected features
    X_selected = m_important.fit_transform(X_train, y_train)

    #printing selected features (get_support() returns bool array of X_train size, if feature is selected - true)
    print('Selected features:')
    print(features.columns.drop(['-- Rodzaj --', '-- Numer --'])[m_important.get_support() == True])
    print("Number of selected features:", np.count_nonzero(m_important.get_support() == True))

    #train test split on already selected (most important) features and
    X_train2, X_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(X_selected, y_train, test_size=0.25)
    #training the model
    clf.fit(X_train2, y_train2)

    #predicting classes using trained model
    y_predicted = clf.predict(X_test2)
    #printing accuracy
    print("Accuracy:", metrics.accuracy_score(y_test2, y_predicted))

classification_wrapper()



#filter method (generic set of methods which do not incorporate a specific machine learning algorithm to find optimal features)
def classification_filter():
    print('Classification filter method')
    #using mutual_info_classif to find feature importances (X - dataframe with features, y - classes)
    f_importances = mutual_info_classif(X,y)

    #printing importances for individual features
    importances = pd.Series(f_importances, features.columns.drop(['-- Rodzaj --', '-- Numer --'])).sort_values()
    print('Feature importances:')
    print(importances)

    #creating 3 different cases
    #1.train test split with original dataset
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
    #2.train test split on features with importance above 0.35
    index2 = np.where(importances > 0.35)[0]
    X2 = X[:, index2]
    X_train2, X_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(X2, y, test_size=0.25)
    #3.train test split on features with importance above 0.55
    index3 = np.where(importances > 0.55)[0]
    X3 = X[:, index3]
    X_train3, X_test3, y_train3, y_test3 = sklearn.model_selection.train_test_split(X3, y, test_size=0.25)

    #initializing classifier for every case, training and predicting classes using trained model
    clf = RandomForestClassifier(n_estimators=100)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    clf2.fit(X_train2, y_train2)
    clf3.fit(X_train3, y_train3)
    y_predicted = clf.predict(X_test)
    y_predicted2 = clf2.predict(X_test2)
    y_predicted3 = clf3.predict(X_test3)

    #printing accuracy
    print("Accuracy for 1 case (original dataset):", metrics.accuracy_score(y_test, y_predicted))
    print("Accuracy for 2 case (> 0.35):", metrics.accuracy_score(y_test2, y_predicted2))
    print("Accuracy for 3 case (> 0.55):", metrics.accuracy_score(y_test3, y_predicted3))

classification_filter()

#another wrapper method
def classification_wrapper2():
    print('Classification 2 wrapper method')
    clf = RandomForestClassifier(n_estimators=100)

    #initializing rfecv (ecursive feature elimination with cross-validation to select the number of features)
    rfecv = RFECV(estimator=clf, cv=3)

    #returns transformed array with selected features
    X_selected = rfecv.fit_transform(X,y)

    #printing optimal number of features for individual prediction and selected features itself
    print("Optimal number of features:", format(rfecv.n_features_))
    print(features.columns.drop(['-- Rodzaj --', '-- Numer --'])[rfecv.get_support() == True])

    #train test split on already selected (most important) features
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_selected, y, test_size=0.25)
    #training the model
    clf.fit(X_train, y_train)

    #predicting classes using trained model
    y_predicted = clf.predict(X_test)

    #printing accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))

classification_wrapper2()


