#linear regression using classes

import numpy as np
import matplotlib.pyplot as plt


class optimize(object):
 
  def __init__(self,X_f,y):                           #x_f is matrix of n features.

    self.X_f = X_f
    self.y = y
    X_norm = self.normalise()
	
    m = np.shape(y)[0]
    X = np.hstack([np.ones((m,1)),X_norm])
    self.X = X
	
  def gettheta(self,alpha,iterations=100,lambda_=0):
  
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
	
    self.theta = self.gradDescent()
    
    return self.theta

  def costfunc(self,theta):
    
    m= np.shape(self.X)[0]
    boundary_eqn = np.dot(self.X,theta)
    predict = self.sigmoid(boundary_eqn)
    lh = np.log(predict)
    lmh = np.log(1- predict)
    J= (-1.0/m)*np.sum(np.dot(self.y.T,lh)+np.dot((1-self.y).T,lmh))
    regular = (self.lambda_/(2.0*m))*(np.sum(np.multiply(theta,theta))- (theta[0,0]**2))
    J = J + regular
    return J

  def gradDescent(self):

    X = self.X
    y = self.y
    print np.shape(y)
    n=np.shape(X)[1]-1
    m= np.shape(X)[0]
    
    theta = np.random.random((n+1,1))
    print np.shape(theta)
    J_vec = np.ones(self.iterations)

    for i in range(0,self.iterations):
    
      boundary_eqn = np.dot(X,theta)
      predict = self.sigmoid(boundary_eqn)
      grad  = (1.0/m)*np.dot(X.T,predict-y)
      grad = grad + (self.lambda_/m)*theta
      grad[0,0] = grad[0,0] - (self.lambda_/m)*theta[0,0]
      theta = theta - self.alpha*grad
      J_vec[i]=self.costfunc(theta)
      
    self.J_vec =J_vec
    return theta

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

  def predict(self,x):                             # X is a vector of n features
      
    x_norm = np.true_divide((x - self.mean),self.stddev)
    x_new = np.hstack([np.ones((np.shape(x_norm)[0],1)),x_norm])
    predt = np.dot(x_new,self.theta)
    predt = np.greater_equal(predt,0.5)
    predt = predt.astype(int)
    
    return predt

  def accuracy(self,x,y):                              # x is a vector of n features
      y_predt = self.predict(x)
      equ = np.equal(y_predt,y)
      n_correct = np.sum(equ.astype(int))
      accuracy = np.true_divide(n_correct*100,np.shape(y)[0])
      
      return accuracy