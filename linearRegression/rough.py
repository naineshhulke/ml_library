# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:46:12 2018

@author: Nainesh
"""

import numpy as np
import linearRegression as logr
import pandas as pd

file_data = pd.read_csv('linearRegression.txt',header=None)
X_data = np.array(file_data.values)

X = X_data[:,0:np.shape(X_data)[1]-1]
y = X_data[:,np.shape(X_data)[1]-1]

[m,n] = np.shape(X)

X_train = X[0:int(0.7*m),:]
y_train = y[0:int(0.7*m)]
X_test = X[int(0.7*m):,:]
y_test = y[int(0.7*m):]

y_train  = np.reshape(y_train,(32,1))
y_test = np.reshape(y_test,(np.shape(y_test)[0],1))
print np.shape(y_train)
stat = logr.optimize(X_train,y_train)
stat.plotJvsno(1,100)

[theta,c] = stat.gettheta(0.4,10,0,17)
print np.shape(theta)

k = stat.accuracy(X_test,y_test)
print ('Accuracy = ',k)
