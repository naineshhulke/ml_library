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

file = pd.read_csv('data/Corners.txt',header=None,sep=' ')

data = np.array(file.values)

X = data[:,0:2]
y = data[:,2:3]

m = np.shape(X)[0]
X_train = X[0:int(0.7*m),:]
y_train = y[0:int(0.7*m)]
X_test = X[int(0.7*m):,:]
y_test = y[int(0.7*m):]

stat = knn.det(X_train,y_train,4)
stat.kval()
print stat.accuracy(X_test,y_test)


