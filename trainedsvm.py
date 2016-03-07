#!/Users/takashi/.pyenv/shims/python


def trainedsvm(svx, svy, svr, ker, x):
    """
        support vector : svx
        trained coefficient : svr
        the label about the support vector : svy
        kernel funcion : ker

    """
    discrim_v = 0.
    for i, svx_i in enumerate(svx):
        discrim_v = discrim_v + svr[i] * svy[i] * ker(svx_i, x)

    if discrim_v >= 0:
        return 1
    else:
        return -1


def test_trainedsvm():
    x = 0
    y = trainedsvm(x)
    assert y == 1
