#!/Users/takashi/.pyenv/shims/python

import readdata
import loss_func
import sag
import write_sv
from functools import partial
import make_kmx
import gausskernel


if __name__ == '__main__':
    print("***PROGRESSS*** :: tarting making the optimum discriminator.")
    fn = "./chronic_kidney_disease/Chronic_Kidney_Disease_full.arff"

    # read the data
    data = readdata.read_data(fn)
    x = data[0]
    y = data[1]

    # the special case of those, which is the kernel
    kmx = make_kmx.make_kmx(x, gausskernel.gausskernel)
    # get the optimum parameter
    opt_r = sag.sag(partial(loss_func.loss_func, lam=1, y=y, k=kmx),
                    partial(loss_func.derivative_loss_func, lam=1, y=y, k=kmx),
                    dim_leng=len(y))
    # find support vector
    svidx = write_sv.find_sv(opt_r)
    write_sv.write_sv(x, y, opt_r, svidx)
    # conserve the parameter and the support vector
    print("fin")
