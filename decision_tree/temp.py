# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:36:46 2019

@author: Nainesh
"""

import numpy as np


a = np.array([[1,'a','b'],[2,'b','c']])
b = []
b.append(a[0].tolist())
b.append(a[1].tolist())
print np.array(b)

print len(a[:,np.shape(a)[1]-1].T)

print '...................'

import numpy as np

x = np.array([1,1,1,2,2,2,5,25,1,1])
unique, counts = np.unique(x, return_counts=True)
counts = np.true_divide(counts**2,5)
print np.asarray((unique, counts)).T
