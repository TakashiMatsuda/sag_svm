#!/Users/takashi/.pyenv/shims/python


def scaling(data):
    """
        Scaling. Make x's average to 0, variance to 1
    """
    return scaled_data


def test_scaling():
    import numpy
    import numpy.randam as rd
    data = [[rd.rand() * j for i in range(5)] for j in range(5)]
    res = scaling(data)
    assert res.average == 0.
