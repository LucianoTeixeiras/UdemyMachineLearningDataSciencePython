# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:21:48 2019

@author: lteixeira
"""

import pandas as pd
base = pd.read_csv('census.csv')

# Dividindo a base de dados em:

# Previsores

previsores = base.iloc[:, 0:14].values

previsores

classe = base.iloc[:, 14].values

classe

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
# labels = labelencoder_previsores.fit_transform(previsores[:, 1])

previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:, 13])

previsores

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)