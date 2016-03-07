#!/Users/takashi/.pyenv/shims/python

import numpy as np


def find_sv():



def write_sv(svx, svy, svr):
    np.save('svx.npy', svx)
    np.save('svy.npy', svy)
    np.save('svr.npy', svy)
    return 0


def test_write_sv():
    assert 0
