#!/Users/takashi/.pyenv/shims/python

import numpy as np
import gausskernel
import readiris
import drawgraph
from functools import partial
import math


def trainedsvm(svx, svy, svr, ker, x):
    """
        support vector : svx
        trained coefficient : svr
        the label about the support vector : svy
        kernel funcion : ker
    """
    discrim_v = 0.

    for i, svx_i in enumerate(svx):
        discrim_v = discrim_v + svr[i] * svy[i] * ker(svx_i, x)

    if discrim_v >= 0:
        return 1
    else:
        return -1


def test_trainedsvm():
    x = 0
    y = trainedsvm(x)
    assert y == 1


if __name__ == "__main__":
    """
        load the stored parameter
    """
    svx = np.load('svx.npy')
    svy = np.load('svy.npy')
    svr = np.load('svr.npy')

    gamma = math.pow(2, -1)
    ker = partial(gausskernel.gausskernel, gamma=math.pow(2, gamma))
    """
       load the test data
    """
    fn_iris = './iris/usingdata.csv'
    data = readiris.readiris(fn_iris)
    x = [l[:2] for l in data[0]]
    y = data[1]
    """
        run the trained svm
    """
    answer = [0 for a in x]
    for i, x_i in enumerate(x):
        answer[i] = trainedsvm(svx, svy, svr, ker, x_i)
    print("answer:")
    print(answer)
    print("teacher")
    print(y)
    drawgraph.drawgraph(x, y, svx, svy, svr, ker)
