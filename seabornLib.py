# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:17:06 2020

@author: shamaun
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\shamaun\Desktop\Datasets\Churn_Modelling.csv")


#distribution plot

sns.distplot(data.CreditScore)
plt.show()

#CredidScore exited v/s CreditScore not exited

sns.distplot(data.CreditScore[data.Exited==0])
sns.distplot(data.CreditScore[data.Exited==1])
plt.legend(['exited','not-exited'])
plt.show()

#Age exited v/s Age not exited
sns.distplot(data.Age[data.Exited==0])
sns.distplot(data.Age[data.Exited==1])
plt.legend(['exited','not-exited'])
plt.show()

#countplot :- count of categorical values

sns.countplot(data.Gender)
plt.yticks(np.arange(0,6000,500))
plt.grid()
plt.show()

sns.countplot(data.Gender[data.Exited==1])
plt.grid()
plt.show()


sns.countplot(data.Geography)
plt.grid()
plt.show()

sns.countplot(data.Geography[data.Exited==1])
plt.grid()
plt.show()

#heatmap :- displays correlation coeeficient

corr = data.corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr,cmap='rainbow',annot=True)
plt.show()


