# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 12:50:25 2025

@author: yfdantas
"""

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv('datasets/iris.csv')
features = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

previsores_teste, previsores_treinamento, classe_teste, classe_treinamento = train_test_split(features, classe, test_size=0.25)


classificador = LinearSVC(dual=False)
classificador.fit(previsores_treinamento, classe_treinamento)


previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

acuracia = accuracy_score(previsoes, classe_teste)
print("Acuracia do modelo: %.2f"%acuracia)

matriz = confusion_matrix(previsoes, classe_teste)
print(matriz)