#!/Users/takashi/.pyenv/shims/python

import numpy as np


def scaling(data):
    """
        Scaling. Make x's average to 0, variance to 1
    """
    scaled_data = np.zeros_like(data)
    sumlist = np.sum(data, axis=0)
    avglist = np.array([d / len(data[0]) for d in sumlist])
    for i, x in enumerate(data):
        scaled_data[i] = np.array([x[j] - avglist[j] for j in range(len(x))])

    print(scaled_data)
    return scaled_data


def test_scaling():
    from numpy import random as rd
    data = [[rd.rand() * j for i in range(5)] for j in range(5)]
    res = scaling(data)
    print(res)
    assert np.sum(res)
