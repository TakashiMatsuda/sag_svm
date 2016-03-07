#!/Users/takashi/.pyenv/shims/python

import numpy as np


def find_sv(r, y):
    svlist = []
    for i, y_i in enumerate(y):
        if (r[i] * y_i) == 0:
            svlist.append(i)

    return svlist


def write_sv(x, y, r, svlist):
    svx = [x[p] for p in svlist]
    svy = [y[p] for p in svlist]
    svr = [r[p] for p in svlist]
    np.save('svx.npy', svx)
    np.save('svy.npy', svy)
    np.save('svr.npy', svr)
    return 0


def test_find_sv():
    assert 0


def test_write_sv():
    assert 0
