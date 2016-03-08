#!/Users/takashi/.pyenv/shims/python

import read_x
import loss_func
import sag
import write_sv
import sys
from functools import partial
import make_kmx
import read_y
import gausskernel


if __name__ == '__main__':
    print("Starting making the optimum discriminator.")
    fn_x = sys.argv[1]
    fn_y = sys.argv[2]

    # read the data
    x = read_x.read_x(fn_x)
    y = read_y.read_y(fn_y)
    # the special case of those, which is the kernel
    kmx = make_kmx.make_kmx(x, gausskernel)
    # get the optimum parameter
    opt_r = sag(partial(loss_func.loss_func, lam=1, y=y, k=kmx),
                partial(loss_func.derivative_loss_func, lam=1, y=y, k=kmx))
    # find support vector
    svidx = write_sv.find_sv(opt_r)
    write_sv.write_sv(x, y, opt_r, svidx)
    # conserve the parameter and the support vector
    print("fin")
