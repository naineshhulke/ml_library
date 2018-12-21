# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:35:33 2018

@author: Nainesh
"""

import numpy as np

def shuffle(a, b):
  assert len(a)== len(b) 
  p = np.random.permutation(len(a))
  return a[p], b[p]


class det(object):
    
    
  def __init__(self,X,y,classes):
      
    X,y = shuffle(X,y) 
    self.X = X
    X = self.normalise()
    self.y = y
    self.classes = classes
  
  def normalise(self):
    
    mean = np.mean(self.X,0)
    self.mean = mean
    stddev = np.std(self.X,0)
    self.stddev = stddev
    X_norm = np.true_divide((self.X - mean),stddev)
    return X_norm

  def kval(self,k=0):
      
    if k==0:
      self.k = self.classes + 1
    else:
      self.k = k
   
  def dist(self,C,D):
    return np.sum((C - D)**2)
    
  def getclass(self,Xt):                # where Xt must be a 3D array
    
    X = self.X
    y = self.y
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    p = np.shape(Xt)[0]
    Xt = np.reshape(Xt,(p,n))
    
    A = np.zeros((m,p))
    for i in range(0,p):
      k = (Xt[i,:] - X )**2
      k = k.sum(1)
 #    k = k.reshape((np.shape(k)[0],1))
      A[:,i] = k
      
    ind = A.argsort(0)
    
    ind = ind[0:self.k,:]
    
    indcls = np.zeros((np.shape(ind)), dtype=int)
    yt = np.zeros((p,1))
    
    for i in range(0,p):
      for j in range(0,self.k):
        indcls[j,i] = int(y[int(ind[j,i]),0])
      temp = (indcls[:,i]).T
      yt[i,0] = np.argmax(np.bincount(temp),0)
      
    return yt

  def accuracy(self,X,y):
    
    X = np.true_divide((X - self.mean),self.stddev)
    yt = self.getclass(X)
    
    k = (yt==y)
    k = k.astype(int)

    return np.sum(k)*100.0/np.shape(y)[0]
    
    


      
      
      
        
    

          
 
    
    



