# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:11:30 2020

@author: shamaun
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data

labels = iris.target

plt.scatter(data[:,0],data[:,1],c=labels)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

plt.scatter(data[:,1],data[:,2],c=labels)
plt.xlabel('sepal width')
plt.ylabel('petal length')
plt.show()

plt.scatter(data[:,2],data[:,3],c=labels)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()


ip = data[:,2:]

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

LR = LogisticRegression()

NB = GaussianNB()

KN = KNeighborsClassifier(n_neighbors=9)

SV = SVC(C=20,kernel='rbf',gamma=50)

LR.fit(ip,labels)

NB.fit(ip,labels)

KN.fit(ip,labels)

SV.fit(ip,labels)

x_min , x_max = ip[:,0].min() , ip[:,0].max()
y_min , y_max = ip[:,1].min() , ip[:,1].max()

xx,yy = np.meshgrid(np.linspace(x_min, x_max),
                    np.linspace(y_min, y_max))

grid = np.c_[xx.ravel(),yy.ravel()]

predLR = LR.predict(grid).reshape(xx.shape)
predNB = NB.predict(grid).reshape(xx.shape)
predKN = KN.predict(grid).reshape(xx.shape)
predSV = SV.predict(grid).reshape(xx.shape)

plt.contourf(xx,yy,predLR)
plt.scatter(data[:,2],data[:,3],c=labels)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Logistic Regression')
plt.show()

plt.contourf(xx,yy,predNB)
plt.scatter(data[:,2],data[:,3],c=labels)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Naive Bayes')
plt.show()


plt.contourf(xx,yy,predKN)
plt.scatter(data[:,2],data[:,3],c=labels)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('KNN')
plt.show()


plt.contourf(xx,yy,predSV)
plt.scatter(data[:,2],data[:,3],c=labels)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('sv')
plt.show()








