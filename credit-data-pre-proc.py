# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
base= pd.read_csv('credit-data.csv')

base.describe()

base.loc[base['age'] < 0]

# Corrigir dados inconsistentes com drop

base.drop('age', 1, inplace=True)

# Corrigir dado inconsistentes apagando os registros com problemas

base.drop(base[base.age < 0].index, inplace=True)

#Corrigndo dados inconsistentes preenchendo com a média

base.mean()

base['age'].mean()

base['age'][base.age > 0 ].mean()

# Ajustando os dados inconsistentes

base.loc[base.age < 0, 'age'] = 40.92

# Ajustando os dados faltantes

pd.isnull(base['age'])

base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classes = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Ajustando escala das variaveis

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)