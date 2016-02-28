#!/Users/takashi/.pyenv/shims/python

import numpy as np
import numpy.random as rd
from functools import partial


def sag(target, derivative):
    """
    Stochastic average decent method
    to minimize the target function
    target: function, temporally partial.
    """
    LENG = 4
    # initial value is a zero vector
    r = np.zeros(LENG)
#    candi_r = np.zeros(LENG)

    y_vec = np.zeros((LENG, LENG))
    # Random dimension to improve specially

    alpha = 0.8
    for ct in range(100):
        ik = (rd.rand() * LENG) % LENG
        y_vec[ik] = derivative(target, ik, r)
        r = np.subtract(r, (alpha / float(LENG)) * np.sum(y_vec, axis=0))

    return r


def parabora10(r, val_min):
    m = np.array([val_min for x in range(len(r))])
    return (np.square(np.subtract(r, m))).sum()

"""
def test_target():
    val_min = 10
    assert parabora10([val_min for x in range(4)], val_min) == 0
"""


def test_sag():
    """
    parabora10に、val_minを部分適用し、rを引っこ抜いた状態の
    関数を与えて
    返り値が、全ての値が10のベクトルであることを確認する。
    10と一致という概念は、とりあえず、9.9から10.1の中に落ちることを確認する。
    """
    sol_list = sag(partial(parabora10, val_min=10))
    for sol in sol_list:
        assert sol > 9.9 and sol < 10.1, \
            "value was odd, should be inside the range"
