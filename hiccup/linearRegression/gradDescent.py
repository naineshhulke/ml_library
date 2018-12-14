import numpy as np
import costfunc as cf



def gradDescent(X,y,alpha,iterations,lambda_):

    n=np.shape(X)[1]-1
    m= np.shape(X)[0]
    
    theta = np.random.random((n+1,1))
    J_vec = np.ones(iterations)

    for i in range(0,iterations):
    
        predt = np.dot(X,theta)-y
        grad = np.dot(X.T,predt)
        theta = theta - alpha*(1.0/m)*grad
        J_vec[i]=cf.costfunc(X,y,theta)
    
    return [theta,J_vec]