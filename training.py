#!/Users/takashi/.pyenv/shims/python

import read_x
import loss_func
import sag
import write_sv
import sys
from functools import partial


if __name__ == '__main__':
    print("Starting making the optimum discriminator.")
    fn_x = sys.argv[1]
    fn_y = sys.argv[2]

    # read the data
    x = read_x.read_x(fn_x)
    y = read_y.read_y(fn_y)
    # the special case of those, which is the kernel
    # 内包表記でkernel matrixを書けないかな.
    # get the optimum parameter
    sag(partial(loss_func.loss_func, lam=1, y=y, k=)
    # conserve the parameter and the support vector
    print("fin")
