# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:01:21 2018

@author: Nainesh
"""

import numpy as np
import neuralNetwork as nn
import pandas as pd

def shuffle(a, b):
  assert len(a)== len(b) 
  p = np.random.permutation(len(a))
  return a[p], b[p]

data = pd.read_csv('data/Moon.txt',header=None,sep=',')

data = np.array(data.values)

X = data[:,0:2]
y = data[:,2:3]

m = np.shape(X)[0]
X,y = shuffle(X,y)
X,y = shuffle(X,y)

X_train = X[0:int(0.7*m),:]
y_train = y[0:int(0.7*m)]
X_test = X[int(0.7*m):,:]
y_test = y[int(0.7*m):]

print y_train[0:20,:]

stat = nn.optimize(X_train,y_train)
stat.parameter([25],4)
Theta = stat.gettheta(,400,1,0)

y_predict = stat.predict(X_test)
print '..............................'

print stat.accuracy(X_test,y_test)


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 25), random_state=1)
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
y_predict = np.reshape(y_predict,(np.shape(y_predict)[0],1))
y = y_predict
k = (y==y_test)
k = k.astype(int)

print np.sum(k)*100/np.shape(y_test)[0]