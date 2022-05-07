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
#proba1
#import pliku z cechami
cechy = pd.read_csv(r"cechy.csv", decimal=",", sep=";")
X = np.array(cechy.drop(cechy.columns[[0, 1]], axis=1))
y = np.array(cechy["-- Rodzaj --"])
#metoda polegająca na wyborze cech PODCZAS klasyfikacji (wybieram cechy z wylosowanych do modelu trenującego)
def klasyfikacja_wrapper():
    print('Klasyfikacja 1 metoda')
    #klasyfikator rf
    clf = RandomForestClassifier(n_estimators=100)
    #inicjalizuje SelectFromModel, można podać jeszcze threshold oprócz klasyfikatora w parametrze
    #threshold=0.05 - wybieram cechy których istotność w predykcji przekracza 0.05 (ja zostawiłam default)
    najwazniejsze = SelectFromModel(clf)
    #model treningowy: X to dataframe tylko z cechami (bez rodzaju bakterii i numeru zdjęcia), y to numery rodzajów
    X_trenujacy, X_testowy, y_trenujacy, y_testowy = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
    #dopasowuje kalkulacje zwraca wybrane cechy
    X_wybrane=najwazniejsze.fit_transform(X_trenujacy, y_trenujacy)
    #wypisuje jakie i ile cech zostało wybranych (get_support() zwraca bool array rozmiaru X_train, jeśli używamy danej cechy w jej miejscu jest true, jeśli nie - jest false)
    print('Wybrane cechy:')
    print(cechy.columns.drop(['-- Rodzaj --', '-- Numer --'])[najwazniejsze.get_support() == True])
    print("Liczba cech:", np.count_nonzero(najwazniejsze.get_support() == True))
    #robię nowy model trenujący z wybranymi już "najlepszymi" cechami
    X_trenujacy2, X_testowy2, y_trenujacy2, y_testowy2 = sklearn.model_selection.train_test_split(X_wybrane, y_trenujacy, test_size=0.25)
    clf.fit(X_trenujacy2, y_trenujacy2)
    #przewidywany rodzaj na podstawie danych testowych z nowego modelu, po odrzuceniu mniej istotnych cech
    y_przwidywany = clf.predict(X_testowy2)
    #accuracy (zgodność dopasowania)
    print("Accuracy:", metrics.accuracy_score(y_testowy2, y_przwidywany))
klasyfikacja_wrapper()



#metoda polegająca na wyborze cech PRZED klasyfikacją (z góry wybieram cechy przekraczające jakiś próg istotności)
def klasyfikacja_filter():
    print('Klasyfikacja 2 metoda')
    #używam mutual_info_classif aby znaleźć istotność poszczególnych cech
    istotnosc = mutual_info_classif(X,y)
    #wypisuje ważność cech (tylko informacyjnie)
    waznosc_cech = pd.Series(istotnosc, cechy.columns.drop(['-- Rodzaj --', '-- Numer --'])).sort_values()
    print('Istotność cech:')
    print(waznosc_cech)
    #tworzę 3 przypadki
    #1.model trenujący na niefiltrowanych cechach
    X_trenujacy, X_testowy, y_trenujacy, y_testowy = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
    #2.model trenujący z cechami o istotności przekraczającej 0.35
    indeksy2 = np.where(istotnosc > 0.35)[0]
    X2 = X[:, indeksy2]
    X_trenujacy2, X_testowy2, y_trenujacy2, y_testowy2 = sklearn.model_selection.train_test_split(X2, y, test_size=0.25)
    # 2.model trenujący z cechami o istotności przekraczającej 0.55
    indeksy3 = np.where(istotnosc > 0.55)[0]
    X3 = X[:, indeksy3]
    X_trenujacy3, X_testowy3, y_trenujacy3, y_testowy3 = sklearn.model_selection.train_test_split(X3, y, test_size=0.25)
    #dla każdego modelu inicjalizuję klasyfikator, trenuję i przewiduję wynik na podstawie części testowej modelu
    clf = RandomForestClassifier(n_estimators=100)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = RandomForestClassifier(n_estimators=100)
    clf.fit(X_trenujacy, y_trenujacy)
    clf2.fit(X_trenujacy2, y_trenujacy2)
    clf3.fit(X_trenujacy3, y_trenujacy3)
    y_przewidywany = clf.predict(X_testowy)
    y_przewidywany2 = clf2.predict(X_testowy2)
    y_przewidywany3 = clf3.predict(X_testowy3)
    #wypisuję accuracy
    print("Accuracy dla nieprzefiltrowanych cech:", metrics.accuracy_score(y_testowy, y_przewidywany))
    print("Accuracy dla cech o istotności powyżej 0.35:", metrics.accuracy_score(y_testowy2, y_przewidywany2))
    print("Accuracy dla cech o istotności powyżej 0.55:", metrics.accuracy_score(y_testowy3, y_przewidywany3))
klasyfikacja_filter()

def klasyfikacja_wrapper2():
    print('Klasyfikacja 3 metoda')
    # klasyfikator rf
    clf = RandomForestClassifier(n_estimators=100)
    rfecv=RFECV(estimator=clf, cv=3)
    X_wybrane=rfecv.fit_transform(X,y)
    print("Optymalna liczba cech:", format(rfecv.n_features_))
    print(cechy.columns.drop(['-- Rodzaj --', '-- Numer --'])[rfecv.get_support() == True])
    X_trenujacy, X_testowy, y_trenujacy, y_testowy=sklearn.model_selection.train_test_split(X_wybrane, y, test_size=0.25)
    clf.fit(X_trenujacy, y_trenujacy)
    y_przwidywany = clf.predict(X_testowy)
    print("Accuracy:", metrics.accuracy_score(y_testowy, y_przwidywany))
klasyfikacja_wrapper2()


