#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:01:16 2023

Multi Layer Perceptron

@author: yurifarod
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

base = pd.read_csv('datasets/iris.csv')
features = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = to_categorical(classe)
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

previsores_teste, previsores_treinamento, classe_teste, classe_treinamento = train_test_split(features, classe_dummy, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 16, activation = 'relu'))

# Na camada de saida a funcao softmax e necessaria qnd se ha mais de uma classe
# Ha tb uma saida para cada classe
classificador.add(Dense(units = 3, activation = 'softmax'))

# Metrica loss e metrica tb alteradas pelas multiplas opcoes
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

#Aqui o treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 35)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)

print(matriz)
