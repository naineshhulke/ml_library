# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:21:05 2018

@author: Nainesh     flatter converts into non-numpy 1D array
"""

import numpy as np
import pandas as pd
"""
x = np.array([[1],[3],[1],[4],['special type']])
print x.dtype
print 'upar dekhno'

print x
print 'Encoding.............'
b = list(set(x.flatten()))
encode = pd.Series(range(0,len(b)),index = b)
Y = (encode[x.flatten()]).values
y = np.reshape(Y,(-1,1))
print y



print 'Decoding........☺...'

decode = pd.Series(b ,index = range(0,len(b)))
y = (decode[y.flatten()]).values
y = np.reshape(y,(-1,1))
print y



"""
"""
a = np.array([[0, 1], [2, 2], [4, 3]])
print (a == a.max(axis=0)[None,:]).astype(int)

"""

from numpy import *
import math
import matplotlib.pyplot as plt
"""
t = linspace(0, 2*math.pi, 400)
a = sin(t)
b = cos(t)
c = a + b

plt.plot(1, 2, 'bo') # plotting t, a separately 
plt.plot(2, 3, 'bo') # plotting t, b separately 
plt.plot(t, c, 'g') # plotting t, c separately 
plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np
 
x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(1, 2, 'b-') 
line1, = ax.plot(2, 2, 'b-') 
 
for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
    
