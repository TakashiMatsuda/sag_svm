#!/Users/takashi/.pyenv/shims/python
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import trainedsvm



def drawgraph(x, y, svx, svy, svr, ker):
    x = np.array(x)
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = np.zeros_like(xx)
    """
        That code below is slow. It is possible to make it fast.
    """
    for i, z_i in enumerate(z[:]):
        for j, z_j in enumerate(z_i[:]):
            z[i, j] = trainedsvm.trainedsvm(svx, svy, svr, ker, xx[i, j])

    print('z')
    print(z)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    plt.show()

    return 0


def test_drawgraph():
    drawgraph(x, y, svx, svy, svr)
    assert 0
