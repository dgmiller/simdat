import numpy as np
import pandas as pd


def eps(n):
    return np.random.normal(size=n)


def causal_network(B, C, n):

    A = 1.414*B + eps(n)
    D = -2.71*C + eps(n)
    Z = .5*B - 3.14*C + eps(n)

    X = 2*A - .3*Z + eps(n)
    W = .3*X + eps(n)
    Y = 3*W - 1.2*Z + 10*D + eps(n)


    df = pd.DataFrame()

    df['A'] = A
    df['B'] = B
    df['C'] = C
    df['D'] = D
    df['W'] = W
    df['X'] = X
    df['Y'] = Y
    df['Z'] = Z

    return df



