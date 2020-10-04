# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 07:28:13 2020

@author: sambit mohapatra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv(r"E:\ENB2012_datanew.csv",header=None)
data.columns=['Relative Compactness',
'Surface Area',
'Wall Area',
'Roof Area',
'Overall Height',
'Orientation',
'Glazing Area',
'Glazing Area Distribution',
'Heating_Load',
'Cooling_Load']

corr = data.corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr,cmap='rainbow',annot=True)
plt.show()

#getting from heatmap that orientation is a non-affecting factor....

ip=data.drop(['Heating_Load','Cooling_Load','Orientation'],axis=1)
#For Testing, This is the First Data Entry of the csv file
un = np.array([0.98,514.5,294,110.25,7.0,0.0,0])
un = un.reshape(1,-1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Y1
op1=data.Heating_Load
xtr1,xts1,ytr1,yts1=train_test_split(ip,op1,test_size=0.2)
model1=LinearRegression()
model1.fit(xtr1, ytr1)
print(model1.score(xts1, yts1))
print("Y1 ",model1.predict(un))
#Y2
op2=data.Cooling_Load #Y2
xtr2,xts2,ytr2,yts2=train_test_split(ip,op2,test_size=0.2) #Y1
model2=LinearRegression()
model2.fit(xtr2, ytr2)
print(model2.score(xts2, yts2))
print("Y2 ",model2.predict(un))