import numpy as np
import normalise
import gradDescent as gD
import plotcurve as pc
import costfunc as cf


def optimize(X_f,y,alpha,iterations=100,lambda_=0):
    
    (m,n)=np.shape(X_f)
    
    [X_norm,mean,stddev] = normalise.normalise(X_f)
    
    X = np.hstack([np.ones((m,1)),X_norm])
    
    [theta,J_vec] = gD.gradDescent(X,y,alpha,iterations,lambda_)
    
    pc.plotcurve(J_vec,iterations)
	
    J_min =cf.costfunc(X,y,theta)
    
    return [mean,stddev,theta,J_min];