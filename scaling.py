#!/Users/takashi/.pyenv/shims/python

import numpy as np
import math


def scaling(data):
    """
        Scaling. Make x's average to 0, variance to 1
    """
    scaled_data = np.zeros_like(data)
    """
        average section
    """
    sumlist = np.sum(data, axis=0)
    avglist = np.array([d / len(data[0]) for d in sumlist])
    for i, x in enumerate(data):
        scaled_data[i] = np.array([x[j] - avglist[j] for j in range(len(x))])

    """
        variance section
    """
    vr = math.sqrt(np.sum(np.square(scaled_data)))
    scaled_data = np.array([x / vr for x in scaled_data])
    print(scaled_data)
    return scaled_data


def test_scaling():
    data = [[i * j for i in range(5)] for j in range(5)]
    res = scaling(data)
    print(res)
    """
        average test
    """
    assert np.sum(res, axis=0)[3] == 0
    """
        variance test
    """
    assert 0
