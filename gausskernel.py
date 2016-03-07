#!/Users/takashi/.pyenv/shims/python

import numpy as np


def gausskernel(x1, x2):
    """
        x1, x2 must be numpy's ndarrays.
    """
    return np.exp(- np.square(np.subtract(x1, x2)) / 2)


def test_gausskernel():
    tx1 = np.array([1, 0])
    tx2 = np.array([1, 0])
    sol = gausskernel(tx1, tx2)
    print("gausskernel: " + str(sol) + " between " + str(tx1) + " and " + str(tx2))
    assert sol == 0
