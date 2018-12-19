# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:24:14 2018

@author: Nainesh
"""

import numpy as np
from math import sqrt as root


class optimize(object):
  
  def __init__(self,X_f,y):                           #x_f is matrix of n features, y has elements in range(0,nofcls)

    self.X = self.normalise(X_f)
    self.y = y
    
  def normalise(self,x):
      
    mean= np.mean(x,0)
    self.mean = mean
    return (x - mean)
    
  def sigmoid(self,x):
    dr = 1 + np.exp(-x)
    sig  = np.true_divide(1,dr)
    return sig

  def parameter(self,nhid,nout):
    
    self.nhid = nhid
    self.nout = nout
    
  def grad(self,theta1,theta2):
    
    X = self.X
    y = self.y
    m=np.shape(X)[0]
      
      #forprop
    x=X.T
    [a3,a2,a1,z3,z2] = self.forprop(x,theta1,theta2)
      
    Y = np.zeros(np.shape(a3))
    for k in range(0,m):
      Y[y[k,0],k] = 1 
      
      #backprop
      
    del3 = a3 - Y
    del2 = np.dot(theta2[:,1:].T,del3)*self.sigmoid(z2)*(1-self.sigmoid(z2))
      
    grad_1 = np.dot(del2,a1.T)
    grad_2 = np.dot(del3,a2.T)
      
    self.grad_1 = (1.0/m)*grad_1
    self.grad_2 = (1.0/m)*grad_2
    
  def gettheta(self,alpha,iterations=100,lambda_=0):
      
    self.alpha=alpha
    self.iterations = iterations
    self.lambda_ = lambda_
    
    self.gradDescent()
    
    return [self.theta1,self.theta2]
    

    
  def random_intialise(self):
     
    einit1 = root(6)/ (root(np.shape(self.X)[1]) + root(self.nhid))
    einit2 = root(6)/ (root(self.nhid) + root(self.nout) )
    
    theta1 = np.random.random((self.nhid,np.shape(self.X)[1]+1))*einit1 -einit1
    theta2 = np.random.random((self.nout,self.nhid+1))*einit2 -einit2
    
    return [theta1,theta2]
    
  def gradDescent(self):
      
    [theta1,theta2] = self.random_intialise()
    alpha = self.alpha
    
    for i in range(0,self.iterations):
      
      self.grad(theta1,theta2)
      theta1 = theta1 - alpha*self.grad_1
      theta2 = theta2 - alpha*self.grad_2
      
     
    self.theta1 = theta1
    self.theta2 = theta2

  def predict(self,X):
      
    
    X = X - self.mean
    x = X.T
    [a3,a2,a1,z3,z2] = self.forprop(x,self.theta1,self.theta2)
  
    ind = a3.argmax(0)
    a3 = np.zeros(np.shape(a3))                                             # some code is written here, use itwhile bvdk
    for k in range(0,np.shape(a3)[1]):
      a3[ind[k],k] = 1
    
    y = a3.argmax(0)
    y = y.reshape((np.shape(y)[0],1))
    
    return y

  def forprop(self,x,theta1,theta2):       # x is a feature 'vector' 
    
    a1 = np.vstack([np.ones((1,np.shape(x)[1])),x])
    z2 = np.dot(theta1,a1)
    a2 = np.vstack([np.ones((1,np.shape(x)[1])),self.sigmoid(z2)])
    z3 = np.dot(theta2,a2)
    a3 = self.sigmoid(z3)
      
    return [a3,a2,a1,z3,z2]
    
    
  def costfunc(self,X,y,theta1,theta2):
      pass
    
    
    