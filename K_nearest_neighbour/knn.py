# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:35:33 2018

@author: Nainesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class det(object):
    
    
  def __init__(self,X,y):
      
    X,y = self.shuffle(X,y) 
    self.X = X
    X = self.normalise()
    self.y = self.encode(y)
    
  def shuffle(a, b):
    assert len(a)== len(b) 
    p = np.random.permutation(len(a))
    return a[p], b[p]
  
  def encode(self,z):
    b = list(set(z.flatten()))
    self.b = b
    encode = pd.Series(range(0,len(b)),index = b)
    Y = (encode[z.flatten()]).values
    y = np.reshape(Y,(-1,1))
    return y
    
  def normalise(self):
    
    mean = np.mean(self.X,0)
    self.mean = mean
    stddev = np.std(self.X,0)
    self.stddev = stddev
    X_norm = np.true_divide((self.X - mean),stddev)
    return X_norm

  
  def dist(self,C,D):
    return np.sum((C - D)**2)
  
    
  def getclass(self,Xt,k=0):                # where Xt must be a 2D array
    
    if k==0:
      self.k = len(np.unique(self.y)) + 1
    else:
      self.k = k
    X = self.X
    y = self.y
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    p = np.shape(Xt)[0]
    Xt = np.reshape(Xt,(p,n))
    k = self.k
    
    A = np.zeros((m,p))
    for i in range(0,p):
      r = (Xt[i,:] - X )**2
      r = r.sum(1)
      A[:,i] = r
      
    ind = A.argsort(0)
    
    ind = ind[0:k,:]
    indcls = np.zeros((np.shape(ind)), dtype=int)
    yt = np.zeros((p,1))
    
    for i in range(0,p):
      for j in range(0,self.k):
        indcls[j,i] = int(y[int(ind[j,i]),0])
      temp = (indcls[:,i]).T
      yt[i,0] = np.argmax(np.bincount(temp),0)
      
    return self.decode(yt)

  def decode(self,z):
      
    decode = pd.Series( self.b ,index = range(0,len(self.b)))
    y = (decode[z.flatten()]).values
    y = np.reshape(y,(-1,1))
    return y

  def accuracy(self,X,y,k=0):
    
    X,y = self.shuffle(X,y)
    X = np.true_divide((X - self.mean),self.stddev)
    yt = self.getclass(X,k)
    
    k = (yt==y)
    k = k.astype(int)

    return np.sum(k)*100.0/np.shape(y)[0]

  def plotKvsacc(self,X,y,lval):
  
    A = map ( lambda i : self.accuracy(X,y,i) , range(0,lval) )
    plt.plot( range(0,lval) , A )
    plt.xlabel('K')
    plt.ylabel('Accuracy %')
    plt.title('K value vs Accuracy')
    plt.show()
    a = np.argmax(A)
    if a==0 : a = len(np.unique(self.y)) + 1
    print ('Suggested k= '+str(a))
    
    


      
      
      
        
    

          
 
    
    



