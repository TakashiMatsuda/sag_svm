#!/Users/takashi/.pyenv/shims/python

import numpy as np


def find_sv(r):
    """
        This function returns the index numbers that
        are not equal to zero, which are 'support vector.'
    """
    svlist = []
    for i, r_i in enumerate(r):
        if r_i != 0:
            svlist.append(i)

    return svlist


def write_sv(x, y, r, svlist, num):
    svx = [x[p] for p in svlist]
    svy = [y[p] for p in svlist]
    svr = [r[p] for p in svlist]
    np.save(str(num)+'-svx.npy', svx)
    np.save(str(num)+'-svy.npy', svy)
    np.save(str(num)+'-svr.npy', svr)
    return 0


def test_find_sv():
    r = np.array([0.4, 0.2, 0])
    sol = find_sv(r)
    assert sol == [0, 1]


def test_write_sv():
    assert 0
