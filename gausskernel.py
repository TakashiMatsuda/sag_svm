#!/Users/takashi/.pyenv/shims/python

import numpy as np

def gausskernel(x1, x2):
    """
        x1, x2 must be numpy's ndarrays.
    """
    return np.exp(- np.square(np.subtract(x1, x2)) / 2)


def test_gausskernel():
    tx1, tx2 = 0
