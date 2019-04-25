import numpy as np
import pandas as pd
from numpy.random import standard_normal
import matplotlib.pyplot as plt


# input data

def add_intercept(X):
    c = np.ones(X.shape[0])
    return np.column_stack((c.T, X))

# functions
def linear_form(X, params):
    """y=Xb"""
    return X.dot(params)


def noisy(X):
    """Additive noise"""
    return X + standard_normal(size=X.shape)


def kz_filter(x, m, k):
    if m%2 == 0:
        raise ValueError("m must be odd")
    z = pd.Series(x)
    for i in range(k):
        z = z.rolling(window=m, min_periods=1, center=True).mean()
    return z


def treatment_effect():
    pass


### END ###
