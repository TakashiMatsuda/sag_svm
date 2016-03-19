#!/Users/takashi/.pyenv/shims/python

import readdata
import training
import numpy as np

N = 5


def splitdata(n, fn):
    data = readdata.read_data(fn)
    x = data[0]
    y = data[1]

    sp_x = [x[i * len(x) / n:
            ((i + 1) * (len(x) / n) - 1)
            if ((i + 1) * (len(x) / n) - 1 > len(x)) else len(x)]
            for i in range(n)]
    sp_y = [y[i * len(y) / n:
            ((i + 1) * (len(y) / n) - 1)
            if ((i + 1) * (len(y) / n) - 1 > len(y)) else len(y)]
            for i in range(n)]
    rlist = []
    for i, v_y in enumerate(sp_y):
        rlist.append(training.training(sp_x[i], v_y))
        print(rlist[i])
        # conserve the parameter and the support vector in opt_r
#        svidx = write_sv.find_sv(opt_r)
#        write_sv.write_sv(x, y, opt_r, svidx)

    # Fin
    print("PROGRESS :: fin")


if __name__ == '__main__':
    fn = "./chronic_kidney_disease/Chronic_Kidney_Disease_full.arff"
    splitdata(2, fn)
