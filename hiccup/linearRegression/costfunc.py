import numpy as np



def costfunc(X,y,theta):
    m = np.shape(X)[0]
    predit = np.dot(X,theta)
    sq_error = np.power((predit-y),2)
    J = (1.0/(2*m))*np.sum(sq_error)
    return J