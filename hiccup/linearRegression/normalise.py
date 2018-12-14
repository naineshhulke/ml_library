import numpy as np


def normalise(X_f):
    mean= np.mean(X_f,0)
    stddev = np.std(X_f,0)
    X_norm = np.true_divide((X_f - mean),stddev)
    return [X_norm,mean,stddev]