#!/Users/takashi/.pyenv/shims/python

import numpy as np
import numpy.random as rd
from functools import partial
import math


def sag(target, derivative, dim_leng, upper_lim, lower_lim):
    """
    Stochastic average decent method
    to minimize the target function
    target: function, temporally partial.
    """
    """
        dim_leng : the number of data
    """
    r = np.random.uniform(2)

    # Random dimension to improve specially
    gra_vec = np.zeros(dim_leng)
    alpha = 0.001
    ik = 0
    for ct in range(1000000):
        ik = int((rd.rand() * dim_leng) // 1)
        gra_vec[ik] = derivative(r=r, i=ik)
        for cnt_r, compo_r in enumerate(r[:]):
            rnw_compo_r = compo_r - (alpha / float(dim_leng) * gra_vec[cnt_r])
            if rnw_compo_r > upper_lim:
                rnw_compo_r = upper_lim
            elif rnw_compo_r < lower_lim:
                rnw_compo_r = lower_lim
            r[cnt_r] = rnw_compo_r
        if ct % 100 == 0:
            print('r')
            print(r)
    print('sag fin')
    return r


"""
この下の2つの関数minus2, di_parabora10は、
centerが最小点の二次関数の微分係数を計算する。

2つの関数に分けてしまうのは不本意なので、
1つにまとめる方法がないかな。
"""


def minus2(a, center):
    return 2 * (a - center)


def div_parabora10(r, i):
    vfunc = np.vectorize(minus2)
    tmp = vfunc(r, 10)
    return tmp.sum()


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
    sol_list = sag(partial(parabora10, val_min=10), div_parabora10, 4, 100, -100)
    for sol in sol_list:
        print(sol)
        assert sol > 9.5 and sol < 10.5, \
            "value was odd, should be inside the range"

"""
def test_sag_range():
    sol_list = sag(partial(parabora10, val_min=10), div_parabora10, 4, 1, 0)
    for sol in sol_list:
        print(sol)
        assert sol <= 1. and sol >= 0., \
            "value was odd, should be inside the range"
"""
