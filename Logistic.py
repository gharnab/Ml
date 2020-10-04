# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:33:23 2020

@author: sambit
"""

"""
Logistic_regression :- Classification
"""
import numpy as np
import matplotlib.pyplot as plt

n_pts = 100
top_x = np.random.normal(10,2,n_pts)
top_y = np.random.normal(10,2,n_pts)

top_reg = np.array((top_x,top_y)).T

bot_x = np.random.normal(5,2,n_pts)
bot_y = np.random.normal(5,2,n_pts)

bot_reg = np.array((bot_x,bot_y)).T

x_train = np.vstack((top_reg,bot_reg))

labels = np.matrix(np.append(np.ones(n_pts),
                             np.zeros(n_pts))).T
plt.scatter(x_train[:n_pts,0],
            x_train[:n_pts,1],color='r')

plt.scatter(x_train[n_pts:,0],
            x_train[n_pts:,1],color='b')
plt.show()


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,labels)

model.predict_proba(np.array([[7,7]]))





