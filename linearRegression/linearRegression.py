#linear regression using classes

import numpy as np
import matplotlib.pyplot as plt

def normalise(X_f):
  mean= np.mean(X_f,0)
  stddev = np.std(X_f,0)
  X_norm = np.true_divide((X_f - mean),stddev)
  return X_norm
	
def costfunc(X,y,theta,lambda_):
    
  m = np.shape(X)[0]
  predit = np.dot(X,theta)
  sq_error = np.power((predit-y),2)
  J = (1.0/(2*m))*np.sum(sq_error)
  regular=(lambda_/(2*m))*(np.sum(np.multiply(theta,theta))- (theta[0,0]**2))
  J = J + regular
  return J
	
def gradDescent(X,y,alpha,iterations,lambda_):

  n=np.shape(X)[1]-1
  m= np.shape(X)[0]
    
  theta = np.random.random((n+1,1))
  J_vec = np.ones(iterations)

  for i in range(0,iterations):
    
    predt = np.dot(X,theta)-y
    grad = np.dot(X.T,predt)
    theta = theta*(1 - (lambda_*alpha)/m) - alpha*(1.0/m)*grad
    J_vec[i]=costfunc(X,y,theta,lambda_)
    
  return [theta,J_vec]



class optimize(object):
 
  def __init__(self,X_f,y):        #x_f is matrix of n features.

    self.X_f = X_f
    self.y = y
    X_norm = normalise(self.X_f)
	
    m = np.shape(y)[0]
    X = np.hstack([np.ones((m,1)),X_norm])
    self.X = X
	
  def gettheta(self,alpha,iterations=100,lambda_=0):
  
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
	
    [theta,J_vec] = gradDescent(self.X,self.y,self.alpha,self.iterations,self.lambda_)
	
    return theta;
	
  def plotJvsno(self,alpha,iterations,lambda_=0):
    
    self.alpha = alpha
    self.iterations = iterations
    self.lambda_ = lambda_
	
    [theta,J_vec] = gradDescent(self.X,self.y,self.alpha,self.iterations,self.lambda_)
	
    plt.plot(np.arange(0,iterations),J_vec)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()	