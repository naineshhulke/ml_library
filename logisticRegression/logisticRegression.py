# LOGISTIC REGRESSION

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


class optimize(object):
 

  def __init__(self,X_f,y):                            #x_f is matrix of n features.

    X_f,y = self.shuffle(X_f,y) 
    self.X_f = X_f
    self.y = self.encode(y)
    self.X = self.normalise()
 

  def encode(self,z):
    b = list(set(z.flatten()))
    self.b = b
    encode = pd.Series(range(0,len(b)),index = b)
    Y = (encode[z.flatten()]).values
    y = np.reshape(Y,(-1,1))
    return y


  def shuffle(self,a, b):

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p] 


  def gettheta(self,alpha,iterations=100,lambda_=0,batsize=0):
  
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
    self.batsize=batsize
	 
    if batsize==0:
      [self.theta,self.c] = self.gradDescent()
    else:
      if int(np.shape(self.X)[0]%batsize) is 0:
        [self.theta,self.c] = self.mgradDescent()
      else:
        print 'Batch size is invalid. Running Batch Gradient Descent....'
        [self.theta,self.c] = self.gradDescent()
    
    return [self.theta,self.c]


  def costfunc(self,theta,c):
    
    m= np.shape(self.X)[0]
    boundary_eqn = np.dot(self.X,theta) + c
    predict = self.sigmoid(boundary_eqn)
    lh = np.log(predict)
    lmh = np.log(1- predict)
    J= (-1.0/m)*np.sum(np.dot(self.y.T,lh)+np.dot((1-self.y).T,lmh))
    regular = (self.lambda_/(2.0*m))*np.sum(np.multiply(theta,theta))
    J = J + regular
    return J


  def gradDescent(self):

    X = self.X
    y = self.y
    n=np.shape(X)[1]
    m= np.shape(X)[0]
    
    theta = np.random.random((n,1))
    c = random.uniform(0,1)
    J_vec = np.ones(self.iterations)

    for i in range(0,self.iterations):
    
      boundary_eqn = np.dot(X,theta)
      predict = self.sigmoid(boundary_eqn + c)
      grad  = (1.0/m) * np.dot( X.T,predict - y )
      grad = grad + (self.lambda_/m)*theta
      theta = theta - self.alpha*grad
      c = c - (self.alpha/m)*np.sum(predict - y)
      J_vec[i]=self.costfunc(theta,c)
      
    self.J_vec =J_vec
    return [theta,c]


  def mgradDescent(self):
    
    n = np.shape(self.X)[1]
    m = np.shape(self.X)[0]
    
    theta = np.random.random((n,1))
    c = random.uniform(0,1)
    
    for j in range(0,self.iterations):
      
      for i in range(0,m,self.batsize):
        X = self.X[i:i+self.batsize,:]
        y = self.y[i:i+self.batsize,:]
        
        boundary_eqn = np.dot(X,theta)
        predict = self.sigmoid(boundary_eqn + c)
        grad  = (1.0/m) * np.dot( X.T,predict - y )
        grad = grad + (self.lambda_/m)*theta
        theta = theta - self.alpha*grad
        c = c - (self.alpha/m)*np.sum(predict - y)
        
    return [theta,c]
    
    
  def normalise(self):
      
    mean= np.mean(self.X_f,0)
    self.mean = mean
    stddev = np.std(self.X_f,0)
    self.stddev = stddev
    X_norm = np.true_divide((self.X_f - mean),stddev)
    return X_norm
	
    
  def plotJvsno(self,alpha,iterations,lambda_=0):
    
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
	
    self.gradDescent()
	
    plt.plot(np.arange(0,iterations),self.J_vec)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()
  

  def sigmoid(self,x):
    dr = 1 + np.exp(-x)
    sig  = np.true_divide(1,dr)
    return sig


  def predict(self,x):                                 # X is a vector of n features
      
    x_ = np.true_divide((x - self.mean),self.stddev)
    predt = np.dot(x_,self.theta) + self.c
    predt = np.greater_equal(predt,0.5)
    predt = predt.astype(int)
    
    return self.decode(predt)


  def decode(self,z):
      
    decode = pd.Series( self.b ,index = range(0,len(self.b)))
    y = (decode[z.flatten()]).values
    y = np.reshape(y,(-1,1))
    return y


  def accuracy(self,x,y):                              # x is a vector of n features
      y_predt = self.predict(x)
      equ = np.equal(y_predt,y)
      n_correct = np.sum(equ.astype(int))
      accuracy = np.true_divide(n_correct*100,np.shape(y)[0])
      
      return accuracy
    
