# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:18:25 2018

@author: Nainesh
"""
"""
a = np.array([[1,2,2,2,2],[1,2,3,1,2,1,1,1,3,2,2,1]] )
counts = np.bincount(a)
print counts
print np.argmax(counts)
"""
"""
import numpy as np

x = np.array([[1,2,22,2,2],[1,1,1,2,2,2,5,25,1,1]])
unique, counts = np.unique(x, return_counts=True)

print np.asarray((unique, counts)).T
"""

import numpy as np
import pandas as pd
import knn as knn
import matplotlib.pyplot as plt

def shuffle(a, b):
  assert len(a)== len(b) 
  p = np.random.permutation(len(a))
  return a[p], b[p]



file = pd.read_csv('corners.txt',header=None,sep=' ')

data = np.array(file.values)

X = data[:,0:2]
y = data[:,2:3]

# X,y = shuffle(X,y)
X,y = shuffle(X,y)
m = np.shape(X)[0]
X_train = X[0:int(0.7*m),:]
y_train = y[0:int(0.7*m)]
X_test = X[int(0.7*m):,:]
y_test = y[int(0.7*m):]

stat = knn.det(X_train,y_train)
y_predt = stat.getclass(X_test)
t = np.reshape(y_train,(np.shape(y_train)[0]))
print stat.accuracy(X_train,y_train)
plt.scatter(X_train[:,0],X_train[:,1],c=t)
plt.show()
t = np.reshape(y_test,(np.shape(y_test)[0]))
plt.scatter(X_test[:,0],X_test[:,1],c=t)
plt.show()
t = np.reshape(y_predt,(np.shape(y_predt)[0]))
plt.scatter(X_test[:,0],X_test[:,1],c=t)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
t = np.reshape(y_train,(np.shape(y_train)[0],))
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,t)
predicted= model.predict(X_test)
t = np.reshape(y_test,(np.shape(y_test)[0],))
k = (predicted==t)
k = k.astype(int)

print np.sum(k)*100.0/np.shape(t)[0]
print np.shape(X_train)[0]
stat.plotKvsacc(X_test,y_test,10)