# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def klasyfikacja():
    # import pliku z cechami
    cechy = pd.read_csv(r"cechy.csv", decimal=",", sep=";")
    # model treningowy: X to dataframe tylko z cechami (bez rodzaju bakterii i numeru zdjęcia), y to numery rodzajów
    X = np.array((cechy.drop(cechy.columns[[0, 1]], axis=1)))
    y = np.array(cechy["-- Rodzaj --"])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
    #klasyfikator rf
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    #sortowanie cech od najbardziej istotnych do najmniej istotnych
    waznosc_cechy = pd.Series(clf.feature_importances_, index=cechy.drop(cechy.columns[[0, 1]], axis=1).columns).sort_values(ascending=False)
    print(waznosc_cechy)
    #przewidywany rodzaj na podstawie danych testowych
    y_przwidywany=clf.predict(X_test)
    #accuracy (zgodność dopasowania)
    print("Accuracy:", metrics.accuracy_score(y_test, y_przwidywany))

klasyfikacja()