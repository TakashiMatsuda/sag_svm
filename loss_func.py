#!/Users/takashi/.pyenv/shims/python

import numpy as np
import copy
import math
import numpy.random as rd


def loss_func(r, lam, y, k):
    """
    This returns the loss function for the SVM.
    r is the parameter of the svm (numpy array, size=n)
    lam is the parameter of the svm (float)
    y is the label of the data (numpy array, size=n)
    k is the kernel matrix of the data (numpy matrix, n * n)
    """
    s1 = r.sum()
    s2 = 0.0

    # TODO: make this code smater and faster using numpy calculation
    # it's not easy.
    for i, y_i in enumerate(y):
        for j, y_j in enumerate(y):
            s2 = s2 + y_i * y_j * r[i] * r[j] * k[i, j]

    s2 = s2 / (2 * lam)
    return (s1 - s2)


def derivative_loss_func(r, lam, y, k, i):
    """
        This returns the derivative of the loss function for the SVM.
        i is the dimension number which differentiates the loss function.
    """
    s1 = 0.
    for j, y_j in enumerate(y):
        if j != i:
            s1 = s1 - y_j * r[j] * k[i, j]

    return 1 - (s1 * y[i] / 2) - (r[i] * y[i] * y[i] * k[i, i])


def test_loss_func():
    r = np.array([0.3, 4])
    lam = 0.4
    y = np.array([1, 0])
    k = np.array([[0.5, 4], [5, 0.1]])
    print(loss_func(r, lam, y, k))

    """
        手計算と合わせる
    """


def test_derivative_loss_func():
    """
    loss_funcの微分係数を計算し、テストする
    """
    r = np.array([0.3, 4])
    lam = 1.
    y = np.array([1, 0])
    k = np.array([[0.5, 4], [5, 0.1]])
    """
        数値微分の値と一致するかどうかでテストを作れる
    """
    def test_param(param):
        h = 0.00001
        for i in range(len(param)):
            tmp = copy.deepcopy(param)
            tmp[i] = tmp[i] + h
            first = loss_func(tmp, lam, y, k)
            second = loss_func(param, lam, y, k)
            numdif = (first - second) / h
            assert abs(numdif - derivative_loss_func(param, lam, y, k, i)) < 2*h
    for i in range(10):
        r = np.array([rd.rand() for x in range(2)])
        test_param(r)
