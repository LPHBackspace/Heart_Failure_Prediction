# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1MWloHCy03Dem42tXiWIMfijK2dFllZ6d

###Preparação da Tabela
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


df_heart_failure = pd.read_csv('Heart failure file path', usecols={'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking',	'time', 'DEATH_EVENT'})
df_heart_failure

df_heart_failure.info()

df_heart_failure.isnull().sum()

# Caso tenha algum valor nulo: df_heart_failure = df_heart_failure.dropna()

df_heart_failure['creatinine_phosphokinase'].value_counts()

df_heart_failure['DEATH_EVENT'].value_counts()

grafio = df_heart_failure['time'].hist(bins=30)

"""### Remoção de Outliers"""


sns.boxplot(df_heart_failure['creatinine_phosphokinase'])

ndf = ''

Q1 = df_heart_failure['creatinine_phosphokinase'].quantile(0.25)
Q3 = df_heart_failure['creatinine_phosphokinase'].quantile(0.75)
IQR = Q3 - Q1
filter = (df_heart_failure['creatinine_phosphokinase'] >= Q1 - 1.5 * IQR) & (df_heart_failure['creatinine_phosphokinase'] <= Q3 + 1.5 *IQR)
ndf = df_heart_failure.loc[filter]
print(ndf.shape)

sns.boxplot(ndf['ejection_fraction'])

Q1 = ndf['ejection_fraction'].quantile(0.25)
Q3 = ndf['ejection_fraction'].quantile(0.75)
IQR = Q3 - Q1
filter = (ndf['ejection_fraction'] >= Q1 - 1.5 * IQR) & (ndf['ejection_fraction'] <= Q3 + 1.5 *IQR)
ndf = ndf.loc[filter]
print(ndf.shape)

sns.boxplot(ndf['platelets'])

Q1 = ndf['platelets'].quantile(0.25)
Q3 = ndf['platelets'].quantile(0.75)
IQR = Q3 - Q1
filter = (ndf['platelets'] >= Q1 - 1.5 * IQR) & (ndf['platelets'] <= Q3 + 1.5 *IQR)
ndf = ndf.loc[filter]
print(ndf.shape)

sns.boxplot(ndf['serum_creatinine'])

Q1 = ndf['serum_creatinine'].quantile(0.25)
Q3 = ndf['serum_creatinine'].quantile(0.75)
IQR = Q3 - Q1
filter = (ndf['serum_creatinine'] >= Q1 - 1.5 * IQR) & (ndf['serum_creatinine'] <= Q3 + 1.5 *IQR)
ndf = ndf.loc[filter]
print(ndf.shape)

sns.boxplot(ndf['serum_sodium'])

Q1 = ndf['serum_sodium'].quantile(0.25)
Q3 = ndf['serum_sodium'].quantile(0.75)
IQR = Q3 - Q1
filter = (ndf['serum_sodium'] >= Q1 - 1.5 * IQR) & (ndf['serum_sodium'] <= Q3 + 1.5 *IQR)
ndf = ndf.loc[filter]
print(ndf.shape)

sns.boxplot(ndf['time'])

ndf

df_heart = ndf

"""### Preparar dados de treino e teste"""

# Separando o X e o Y
features = df_heart.drop(['DEATH_EVENT'], axis=1) # Caso tenha um ID, também é necessário dropa-los aqui
x = features

y = df_heart[['DEATH_EVENT']]

df_heart['DEATH_EVENT'].value_counts()

# Under Sampling
#from imblearn.under_sampling import RandomUnderSampler
#rus = RandomUnderSampler(random_state=0)
#x_resampled, y_resampled = rus.fit_resample(x, y)

# Over Sampling


x_resampled, y_resampled = SMOTE().fit_resample(x, y)

y_resampled['DEATH_EVENT'].value_counts()

# Dividindo dados para treino e dados para teste


x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, 
                                                    test_size = 0.2, 
                                                    random_state = 42)

# Instaciando o objeto scaler


scaler = MinMaxScaler()

# Fit + transform no conjunto de treino
# Utilizando explicitamente as colunas de ambos os lados força que o
# resultado da normalização ainda seja o dataframe (muito mais facil de manipular) e não um numpy array
x_train[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']] = scaler.fit_transform(x_train[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']])

# Agora utilizando o scaler no conjunto de teste
# Utilizar apenas o transform, pois fit é só no conjunto de treino,]
# o conjunto de teste é utilizado para medir a capacidade de generalização do modelo no mundo real (dados não vistos)
# então faz sentido que a mesma normalização treinada e submetida ao conjunto de treino seja apenas aplicada no de teste
x_test[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']] = scaler.transform(x_test[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']])

"""### Regressão Logistica"""



modelo_lr = LogisticRegression(solver='lbfgs', max_iter=500)
modelo_lr.fit(x_train, y_train.squeeze())

print('Acertividade treino: ', modelo_lr.score(x_train, y_train))
print('Acertividade teste: ', modelo_lr.score(x_test, y_test.squeeze()))



y_pred = modelo_lr.predict(x_test)
print( 'Revocação: ', recall_score( y_test, y_pred ))

# Matriz de confusão


matriz_confusao = confusion_matrix(y_test, y_pred, normalize = 'true')

display = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao, display_labels = ['Não Morreu', 'Morreu'])

display.plot()
plt.show()
