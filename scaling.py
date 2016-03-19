#!/Users/takashi/.pyenv/shims/python

import numpy as np
import math


def scaling(data):
    """
        Scaling. Make x's average to 0, variance to 1
        => CHANGED. Divide by normal deviation
    """
    print("input:")
    print(data)
    scaled_data = np.zeros_like(data)
    """
        average section
    """
    sumlist = np.sum(data, axis=0)
    avglist = np.array([d / len(data) for d in sumlist])
    print("avglist:")
    print(avglist)
    for i, x in enumerate(data):
        scaled_data[i] = np.array([x[j] - avglist[j] for j in range(len(x))])

    """
        variance section
    """
    vrlist = np.var(scaled_data, axis=0)
    print("average=0 data:")
    print(scaled_data)
    return np.divide(scaled_data, vrlist)
    """
    vr = (math.sqrt(np.sum(np.square(scaled_data)))) / len(data)
    scaled_data = np.array([x / vr for x in scaled_data])
    """
#    print(scaled_data)
#    return scaled_data


def test_scaling():
    """
        TODO: More Precise Test is necessary
    """
    data = [[(i+1) * (j+1) for i in range(5)] for j in range(2)]
    res = scaling(data)
    print("res:")
    print(res)
    """
        average test
    """
    assert np.sum(res, axis=0)[1] == 0
    """
        variance test
    """
    assert np.var(res, axis=0)[1] == 1
