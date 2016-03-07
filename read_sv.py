#!/Users/takashi/.pyenv/shims/python

import numpy as np


def read_sv():
    """
        This code reads svx, svy, svr from '*.npy'
        and returns the tuple composed of them.
    """
    svx = np.load('svx.npy')
    svy = np.load('svy.npy')
    svr = np.load('svr.npy')

    return (svx, svy, svr)


def test_read_sv():
    """
        not yet implemented
    """
    assert 0
