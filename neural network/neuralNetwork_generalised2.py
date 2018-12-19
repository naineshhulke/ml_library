# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:56:43 2018

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

  def parameter(self,units,nout):               # units is a list- [no of unit in hidden layers]
      
      nlayer = len(units) + 2
      units.insert(0,int(np.shape(self.X)[1]))
      units.insert(0,int(0))
      units.append(nout)
      self.nlayer = nlayer
      self.units = units                    # now, unit is a list - [ int(0) , no_of unit in layer 1, no of unit in layer 2,]
    
  def gettheta(self,alpha,iterations=100,lambda_=0):
      
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
    
    self.gradDescent()
    
    return self.Theta

  def random_theta(self,i):
    
    units = self.units
    einit = root(6) / ( root( units[i]) + root(units[i+1]) )
    theta = np.random.random(( units[i+1],units[i]+1) )*2*einit - einit
    return theta
    

  def random_intialize(self):
    
                                                    #for sake of convention of notation, Theta[0] will contain garbage value
    Theta = [0]
    
    for i in range(1,self.nlayer):   # generating theta1 as Theta[1]  and so on , so on . . . .    
      k = self.random_theta(i)
      Theta.append(k)
      
    return Theta


  def forprop(self,x,Theta):
    
    A = [0]
    Z = [0]
    m = np.shape(x)[1]

    Z.append(x)
    k = np.vstack([np.ones((1,m)),Z[1]])
    A.append(k)
    
    for i in range(2,self.nlayer):
      
      k = np.dot(Theta[i-1],A[i-1])
      Z.append(k)                                        # Z[i] is formed
      k = np.vstack([np.ones((1,m)),self.sigmoid(Z[i])])
      A.append(k)                                        # A[i] is formed
      
    k = np.dot(Theta[self.nlayer-1],A[self.nlayer-1])
    Z.append(k)
    k = self.sigmoid(Z[self.nlayer])
    A.append(k)
    
    return [A,Z]
    
  def grad(self,Theta):
    
    X = self.X
    y = self.y
    nlayer = self.nlayer
    m=np.shape(X)[0]
    
    #forprop
    x=X.T
    [A,Z] = self.forprop(x,Theta)
    
    Y = np.zeros(np.shape(A[nlayer]))
    for k in range(0,m):
      Y[y[k,0],k] = 1

    #backprop
    Del = []
    k = A[nlayer] - Y
    Del.append(k)
    
    for i in range(nlayer-1,1,-1):
      k = np.dot(Theta[i][:,1:].T,Del[nlayer-1-i])
      k = k*self.sigmoid(Z[i])*(1-self.sigmoid(Z[i]))
      Del.append(k)
      
    Del.append(0)
    Del.append(0)
    Del = list(reversed(Del))
    
    Grad = [0]
    for i in range(1,nlayer):
      k = (1.0/m)*np.dot(Del[i+1],A[i].T)
      Grad.append(k)
    
    self.Grad = Grad
    
    

  def gradDescent(self):
    
    Theta = self.random_intialize()
    alpha = self.alpha
    nlayer = self.nlayer
    
    for i in range(0,self.iterations):
        
      self.grad(Theta)
      for i in range(1,nlayer):
        Theta[i] = Theta[i] - alpha*self.Grad[i]
    
    self.Theta = Theta
    
  def predict(self,x):
      
      nlayer = self.nlayer
      x = x - self.mean
      x = x.T
      [A,Z] = self.forprop(x,self.Theta)
      
      ind = A[nlayer].argmax(0)
      A[nlayer] = 0*A[nlayer]
      for k in range(0,np.shape(A[nlayer])[1]):
        A[nlayer][ind[k],k] = 1
      
      y = A[nlayer].argmax(0)
      y = y.reshape((np.shape(y)[0],1))
      
      return y
      
        