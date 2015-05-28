# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:51:36 2015

@author: ayanami
"""

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
Xtr,Xts,Ytr,Yts = train_test_split(X,y,test_size = 0.5)

from sklearn.metrics import accuracy_score


import model as wr
model = wr.models.rf
azure = wr.Azure_model(login = "sasha_panin@mail.ru",
               password="243414",
               model=model)

azure.fit(X,y)
Ypred = azure.predict(Xts)["Scored Labels"]

print accuracy_score(Yts,Ypred)
