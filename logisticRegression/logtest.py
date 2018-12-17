# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:46:12 2018

@author: Nainesh
"""

import numpy as np
import logisticRegression as logr
import pandas as pd

file_data = pd.read_csv('LogisticRegressionData2.txt',header=None)
X_data = np.array(file_data.values)

X = X_data[:,0:np.shape(X_data)[1]-1]
y = X_data[:,np.shape(X_data)[1]-1]

[m,n] = np.shape(X)

X_train = X[0:int(0.7*m),:]
y_train = y[0:int(0.7*m)]
X_test = X[int(0.7*m):,:]
y_test = y[int(0.7*m):]


y_test = np.reshape(y_test,(np.shape(y_test)[0],1))
y_train = np.reshape(y_train,(np.shape(y_train)[0],1))
print np.shape(y_train)


stat = logr.optimize(X_train,y_train)
stat.plotJvsno(3,100)
y_test = np.reshape(y_test,(np.shape(y_test)[0],1))
theta = stat.gettheta(1,100)


k = stat.accuracy(X_test,y_test)
print ('Accuracy = ',k)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_ = model.predict(X_test)
y_ = y_.reshape((np.shape(y_test)[0],1))
equ = np.equal(y_,y_test)
n_correct = np.sum(equ.astype(int))
accuracy = np.true_divide(n_correct*100,np.shape(y_test)[0])
print accuracy