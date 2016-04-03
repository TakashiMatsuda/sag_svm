#!/Users/takashi/.pyenv/shims/python

import readdata
import loss_func
import sag
import write_sv
from functools import partial
import make_kmx
import gausskernel
import scaling
import numpy as np
import readiris


def training(x, y):
    # calculating the gausskernel (for the experiment)
    kmx = make_kmx.make_kmx(x, gausskernel.gausskernel)

    # get the optimum parameter
    opt_r = sag.sag(partial(loss_func.loss_func, lam=1, y=y, k=kmx),
                    partial(loss_func.derivative_loss_func, lam=1, y=y, k=kmx),
                    dim_leng=len(y))

    return opt_r


if __name__ == '__main__':
    print("PROGRESS :: starting making the optimum discriminator.")
#    fn = "./chronic_kidney_disease/Chronic_Kidney_Disease_full.arff"
#    np.set_printoptions(threshold='nan')
    # read the data
#    data = readdata.read_data(fn)
    fn_iris = "./iris/usingdata.csv"
    data = readiris.readiris(fn_iris)
    x = data[0]
    y = data[1]

    print("x: ")
    print(x)
    # scaling x
    x = scaling.scaling(x)
    print("scaled x:")
    print(x)

    # calculating the gausskernel (for the experiment)

    kmx = make_kmx.make_kmx(x, gausskernel.gausskernel)

    # get the optimum parameter
    opt_r = sag.sag(partial(loss_func.loss_func, lam=1, y=y, k=kmx),
                    partial(loss_func.derivative_loss_func, lam=1, y=y, k=kmx),
                    dim_leng=len(y))

    print(opt_r)  # TODO: delete this line

    # conserve the parameter and the support vector in opt_r
    svidx = write_sv.find_sv(opt_r)
    write_sv.write_sv(x, y, opt_r, svidx)

    # Fin
    print("PROGRESS :: fin")
