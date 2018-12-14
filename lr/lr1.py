
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lrcostfunction(X,y,theta):
    m = np.shape(X)[0]
    predit = np.dot(X,theta)
    sq_error = np.power((predit-y),2)
    J = (1.0/(2*m))*np.sum(sq_error)
    return J
    


# linear regression

file_data = pd.read_csv("F:/abc.csv",header=None)
X_data = np.array(file_data.values)

# taking out features
s=np.shape(X_data)
X_f = X_data[:,0:s[1]-1]
y = X_data[:,s[1]-1]


#normalising features
mean= np.mean(X_f,0)
stddev = np.std(X_f,0)
X_norm = np.true_divide((X_f - mean),stddev)

#creating data matrix
X = np.hstack([np.ones((s[0],1)),X_norm])


#gradient_descent
n=np.shape(X)[1]-1
m= np.shape(X)[0]
alpha=0.05
iterations=100
init_theta = np.ones((n+1,1))
theta = init_theta
J_vec = np.ones(iterations)

for i in range(0,100):
    predt = np.dot(X,theta)-y
    grad = np.dot(X.T,predt)
    theta = theta - alpha*(1.0/m)*grad
    J_vec[i]=lrcostfunction(X,y,theta)

plt.plot(np.arange(0,100),J_vec)
plt.show()
    
  

