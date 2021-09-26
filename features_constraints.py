# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:49:24 2021

@author: Felipe Navarro
"""

#%%
""" 
Preparando o problema:
Consistirá em uma corrida de carros em linha reta, onde a aceleração e a velocidade
inicial mudarão 3 vezes.
A IA deverá dizer qual dos carros (0 ou 1) chegou mais longe, empates serão
eliminados.


Ambos partirão do mesmo ponto inicial.
As velocidades e as acelerações mudarão simultaneamente para ambos os carros.
As velocidades iniciais de um trecho não terão qualquer relação com a velocidades
finais dos trechos anteriores.
Os 4 trechos terão duração de 10 segundos.

Função horária do MRUV:
    S=So+Vot+at2/2


"""



#%%

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import xgboost as xgb

#%%

t= 10
size = 10000

v0 = np.random.uniform(low=60, high=80, size=(size,2))
a0 = np.random.uniform(low=1, high=5, size=(size,2))
s0 = v0*t + a0*(t**2)/2

v1 = np.random.uniform(low=60, high=80, size=(size,2))
a1 = np.random.uniform(low=1, high=5, size=(size,2))
s1 = v1*t + a1*(t**2)/2

v2 = np.random.uniform(low=60, high=80, size=(size,2))
a2 = np.random.uniform(low=1, high=5, size=(size,2))
s2 = v2*t + a2*(t**2)/2

v3 = np.random.uniform(low=60, high=80, size=(size,2))
a3 = np.random.uniform(low=1, high=5, size=(size,2))
s3 = v3*t + a3*(t**2)/2

gab = s0 + s1 + s2 + s3

if sum(np.where(gab[:,0] == gab[:,1], 1, 0 )):
    import sys
    sys.exit()

y = np.where(gab[:,0]>gab[:,1], 1, 0 )

#%%

df = pd.DataFrame(data = (np.column_stack((v0,v1,v2,v3,a0,a1,a2,a3))),
                  columns = ['v00','v01','v10','v11','v20','v21','v30','v31',
                             'a00','a01','a10','a11','a20','a21','a30','a31',
                             ]
                  )


X_train = df.loc[:7499,:].values
y_train = y[:7500]

X_test = df.loc[7500:,:].values
y_test = y[7500:]


#%%

params0 = {
  'colsample_bynode': 0.8,
  # 'learning_rate': 1,
  'max_depth': 5,
  'num_parallel_tree': 100,
  'objective': 'binary:logistic',
  'subsample': 0.8,
  # 'tree_method': 'gpu_hist'
}

params1 = {
  'colsample_bynode': 0.8,
  # 'learning_rate': 1,
  'max_depth': 5,
  'num_parallel_tree': 100,
  'objective': 'binary:logistic',
  'subsample': 0.8,
  # 'tree_method': 'gpu_hist',
  'interaction_constraints':[list(range(8)), list(range(8,16))]
}

params2 = {
  'colsample_bynode': 0.8,
  # 'learning_rate': 1,
  'max_depth': 5,
  'num_parallel_tree': 100,
  'objective': 'binary:logistic',
  'subsample': 0.8,
  # 'tree_method': 'gpu_hist',
  'interaction_constraints':[[i,i+1] for i in range(0,16,2)]
}

seed = np.random.randint(1000000)

xgb_0 = xgb.XGBRFClassifier(**params0, random_state=seed).fit(X_train, y_train)

xgb_1 = xgb.XGBRFClassifier(**params1, random_state=seed).fit(X_train, y_train)

xgb_2 = xgb.XGBRFClassifier(**params2, random_state=seed).fit(X_train, y_train)

predict0 = xgb_0.predict(X_test)
predict1 = xgb_1.predict(X_test)
predict2 = xgb_2.predict(X_test)


#%%
from sklearn.metrics import accuracy_score

accuracy0 = accuracy_score(y_test, predict0)
accuracy1 = accuracy_score(y_test, predict1)
accuracy2 = accuracy_score(y_test, predict2)

print('A acuracia SEM isolamento de variaveis foi de:', accuracy0)
print('A acuracia COM isolamento de variaveis foi de:', accuracy1)
print('A acuracia COM isolamento maior  de variaveis foi de:', accuracy2)

#%%












