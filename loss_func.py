#!/Users/takashi/.pyenv/shims/python

import numpy as np


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
    s = 0.
    for j, y_j in enumerate(y):
        s = s + y_j * r[j] * k[i, j]

    return (1 - s * y[i] / 2)


def test_loss_func():
    r = np.array([0.3, 4])
    lam = 0.4
    y = np.array([1, 0])
    k = np.array([[0.5, 4], [5, 0.1]])
    print(loss_func(r, lam, y, k))
    assert 0


def test_derivative_loss_func():
    """
    loss_funcの微分係数を計算し、テストする
    """
    r = np.array([0.3, 4])
    lam = 1.
    y = np.array([1, 0])
    k = np.array([[0.5, 4], [5, 0.1]])
    # TODO:
    # Make the codo to check that r, y, k has the same dimension.
    print(derivative_loss_func(r, lam, y, k, 0))
    assert 0
