# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:58:31 2018

@author: Nainesh
"""


import numpy as np


class clusterit(object):
    
  def __init__(self,X):

    self.X = X

    
  def getcluster(self,K,iterations = 100):
      
    U = self.X[ np.random.randint( np.shape(self.X)[0] , size = K ) , : ]
    for i in range(0,iterations):
      C = self.cluster_assign(U)
      U = self.move_centroid(C)
    self.U = U
    self.C = self.cluster_assign(U)
    self.labels = self.C
    print ( 'Total Clusters found : ' + str(len(np.unique(self.C))) )
    
    
  def cluster_assign(self,U):
     
    C = (((self.X[:,np.newaxis,:]-U)**2).sum(2)).argmin(1)                                                # C is a 1 D numpy array
    return C

    
  def move_centroid(self,C):
      
    centroids = np.unique(C)
    U = np.array(map (lambda k: self.X[np.ravel(np.where(C==k)).astype('int') , : ].mean(0) , centroids))  # D is a 2 D numpy array
    return U


  def class_set(self,k):
      
    return self.X[np.ravel(np.where(self.C==k)).astype('int') , : ]