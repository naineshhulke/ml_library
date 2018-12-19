# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:36:38 2018

@author: Nainesh
"""
import numpy as np
import neuralNetwork as nn
import pandas as pd
import scipy.io as sio



def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]




k = sio.loadmat('ex4data1.mat')
X = np.array(k['X'])
y = np.array(k['y'])

print 
for x in range(0,np.shape(y)[0]):
    if y[x]==10:
        y[x]=0
 # [X,y] = np.random.shuffle([X,y]) 
[X,y]=unison_shuffled_copies(X,y)

[m,n] = np.shape(X)

X_train = X[0:int(0.7*m),:]
y_train = y[0:int(0.7*m)]
X_test = X[int(0.7*m):,:]
y_test = y[int(0.7*m):]



stat = nn.optimize(X_train,y_train)
stat.parameter(25,10)
[theta1,theta2] = stat.gettheta(4.5,50)

y_predict = stat.predict(X_test)
print y_test
print '..............................'

k = (y_predict==y_test)
k = k.astype(int)

print np.sum(k)*100/np.shape(y_test)[0]