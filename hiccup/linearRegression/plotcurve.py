
import numpy as np
import matplotlib.pyplot as plt

def plotcurve(J_vec,iterations):

    plt.plot(np.arange(0,100),J_vec)
    plt.show()