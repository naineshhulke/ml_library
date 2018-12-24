# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:59:22 2018

@author: Nainesh
"""

import scipy.io as sio
import k_means as km
import numpy as np
import matplotlib.pyplot as plt


k = sio.loadmat('ex7data1.mat')

X = np.array(k['X'])

stat = km.clusterit(X)
stat.getcluster(20,100)


plt.scatter(X[:,0],X[:,1],c=stat.C)
plt.show()

