#!/Users/takashi/.pyenv/shims/python

import numpy as np
import gausskernel


def make_kmx(x_vec, ker):
    kmx = np.array([[ker(x_i, x_j) for x_j in x_vec] for x_i in x_vec])
    print("kmx:")
    print(kmx)
    return kmx


def test_make_kmx():
    x_vec = [[1. for i in range(3)] for j in range(5)]
    sol = make_kmx(x_vec, gausskernel.gausskernel)
    print('sol : ' + str(sol))
    assert sol[2, 2] == 1
