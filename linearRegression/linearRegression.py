#linear regression using classes

import numpy as np
import matplotlib.pyplot as plt
import random


class optimize(object):
 
  def __init__(self,X_f,y):                           #x_f is matrix of n features.

    self.X_f = X_f
    self.y = y
    self.X = self.normalise()
	
  def gettheta(self,alpha,iterations=100,lambda_=0,batsize=0):
  
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
    self.batsize = batsize
    
    if batsize == 0:
      [self.theta,self.c] = self.gradDescent()
    elif int(np.shape(self.X)[0]%batsize) is not 0:
      print 'Batch size is invalid. Running Batch Gradient Descent....'
      [self.theta,self.c] = self.gradDescent()
    else:
      [self.theta,self.c] = self.mgradDescent()
    
    return [self.theta,self.c]

  def costfunc(self,theta,c):
    
    m = np.shape(self.X)[0]
    predit = np.dot(self.X,theta) + c
    sq_error = np.power((predit-self.y),2)
    J = (1.0/(2*m))*np.sum(sq_error)
    regular = (self.lambda_/(2*m))*np.sum(np.multiply(theta,theta))
    J = J + regular
    return J

  def gradDescent(self):

    n = np.shape(self.X)[1]
    m = np.shape(self.X)[0]
    
    theta = np.random.random((n,1))
    c = random.uniform(0,1)
    J_vec = np.ones(self.iterations)

    for i in range(0,self.iterations):
    
      predt = np.dot(self.X,theta) + (c- self.y) 
      grad = np.dot((self.X).T,predt)
      c = c - (self.alpha/m)*np.sum(predt)
      theta = theta*(1 - (self.lambda_*self.alpha)/m) - grad*((self.alpha*1.0)/m)
      J_vec[i] = self.costfunc(theta,c)
    
    self.J_vec = J_vec
    return [theta,c]

  def mgradDescent(self):
      
    n = np.shape(self.X)[1]
    m = np.shape(self.X)[0]
    
    theta = np.random.random((n,1))
    c = random.uniform(0,1)
    
    for j in range(0,self.iterations):
        
      for i in range(0,m,self.batsize):
        a = self.X[i:self.batsize,:]
        b = self.y[i:self.batsize,:]
            
        predt = np.dot(a,theta) + (c - b) 
        grad = np.dot((a).T,predt)
        c = c - (self.alpha/m)*np.sum(predt)
        theta = theta*(1 - (self.lambda_*self.alpha)/m) - grad*((self.alpha*1.0)/m)
        
    return [theta,c]
            

  def normalise(self):
      
    mean = np.mean(self.X_f,0)
    self.mean = mean
    stddev = np.std(self.X_f,0)
    self.stddev = stddev
    X_norm = np.true_divide((self.X_f - mean),stddev)
    return X_norm
	
  def plotJvsno(self,alpha,iterations,lambda_=0):
                                                                   # this functin for only batch Grad descent
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
	
    self.gradDescent()
	
    plt.plot(np.arange(0,iterations),self.J_vec)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()
    
  def predict(self,x):                             # X is a vector of n features
      
    x_ = np.true_divide((x - self.mean),self.stddev)
    y = np.dot(x_,self.theta) + self.c
    return y

  def accuracy(self,x,y):                              # x is a vector of n features
      y_predt = self.predict(x)
      error  = (100*(y_predt - y))/y
      error_mean= np.mean(error)
      
      return (100 - error_mean)
