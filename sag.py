#!/Users/takashi/.pyenv/shims/python

import numpy as np
import numpy.random as rd
from functools import partial


def sag(target, derivative, dim_leng):
    """
    Stochastic average decent method
    to minimize the target function
    target: function, temporally partial.
    """

    # initial value is a zero vector
    # This may makes a strange matter.
#    r = np.zeros(dim_leng)
    # improve version
    r = np.array([rd.rand() for x in range(dim_leng)])
#    candi_r = np.zeros(LENG)

    gra_vec = np.zeros((dim_leng, dim_leng))
    # Random dimension to improve specially

    alpha = 0.1
    ik = 0
    for ct in range(2):
        ik = int((rd.rand() * dim_leng) // 1)
        gra_vec[ik] = derivative(r=r, i=ik)
#        gra_vec[ik] = derivative(target, ik, r)

# this line makes same value's vector, that is strange
        r = np.subtract(r, (alpha / float(dim_leng)) * np.sum(gra_vec, axis=0))
# gra_vec is strange, probably due to derivative_loss_func

        print('ik: ' + str(ik))
        print('r:  ' + str(r))
        print('gra_vec:  ' + str(gra_vec))
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
    sol_list = sag(partial(parabora10, val_min=10), div_parabora10, 4)
    for sol in sol_list:
        print(sol)
        assert sol > 9.9 and sol < 10.1, \
            "value was odd, should be inside the range"
