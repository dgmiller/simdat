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


#def generate_data(X, params, epsilon):
#    X = add_intercept(X)
#
#    # same as Y = X.dot(params) + epsilson
#    #Y = noisy(linear_form(X,params))
#    Y = noisy(linear_form(X,params))
#
#    dataset = {
#            'X': X,
#            'Y': Y
#    }
#
#    unobserved = {
#            'params': params,
#            'noise': epsilon
#    }
#
#    return dataset, unobserved

def kz_filter(x, m, k):
    if m%2 == 0:
        raise ValueError("m must be odd")
    z = pd.Series(x)
    for i in range(k):
        z = z.rolling(window=m, min_periods=1, center=True).mean()
    return z

def visualize():
    """
    See the dataset.

    """
    n = 100
    #xdata = standard_normal(size=(n,1))
    ksi = np.random.randn(n+1)
    xdata = np.linspace(-1,1,n+1)
    b = np.array([1, 1])
    #dataset, unobserved = generate_data(X*ksi, b, eps)

    X = add_intercept(xdata)
    Y = linear_form(X,b)
    V = noisy(Y)
    betas = np.linalg.solve(X.T.dot(X), X.T.dot(V))
    yhat = X.dot(betas)

    plt.figure(figsize=(7,7))
    plt.plot(X[:,1], Y, color='grey', alpha=.5)
    plt.scatter(X[:,1], Y, color='grey', marker='.', alpha=.5)
    
    plt.scatter(X[:,1], V, c='orange', marker='.', alpha=.5)
    plt.plot(X[:,1], yhat, color='orange', alpha=.7)

    m = 7
    k = 8
    Vkz = kz_filter(V, m, k)
    vbetas = np.linalg.solve(X.T.dot(X), X.T.dot(Vkz))
    vhat = X.dot(vbetas)

    plt.scatter(X[:,1], kz_filter(V, m, k), c='purple', marker='.', alpha=.3)
    plt.plot(X[:,1], vhat, color='blue')


    plt.axis('equal')
    plt.show()

    step = np.linspace(-10,10,1001)
    u = np.sin(step) + np.random.randn(1001)
    plt.figure(figsize=(12,6))
    plt.plot(step, np.sin(step), color='blue')
    plt.scatter(step, u, c='orange', marker='.', alpha=.5)
    plt.scatter(step, kz_filter(u, 75, 8), c='purple', marker='.', alpha=.3)
    plt.show()

visualize()


### END ###
