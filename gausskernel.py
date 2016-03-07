#!/Users/takashi/.pyenv/shims/python

import numpy as np
import math


def gausskernel(x1, x2):
    """
        x1, x2 must be numpy's ndarrays.
        This code returns a gaussian kernel value between x1 and x2
    """
    subtr = np.subtract(x1, x2)
    return math.exp(- np.inner(subtr, subtr) / 2)


def test_gausskernel():
    tx1 = np.array([1, 0])
    tx2 = np.array([1, 0])
    sol = gausskernel(tx1, tx2)
    print("gausskernel: " + str(sol) +
          " between " + str(tx1) + " and " + str(tx2))

    assert sol == 1
