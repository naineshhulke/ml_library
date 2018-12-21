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

a = [1,2,3]
b = [1,1,1]

a = map(lambda x,y:x-y,a,b)
print a