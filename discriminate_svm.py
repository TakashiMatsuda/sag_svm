#!/Users/takashi/.pyenv/shims/python

import read_sv
import trainedsvm
import gausskernel
import sys
import read_x
import write_y


if __name__ == '__main__':
    params = read_sv.read_sv()
    x = read_x.read_x(sys.argv[1])
    answer = trainedsvm.trainedsvm(params[0], params[1], params[2],
                                   gausskernel.gausskernel, x)
    write_y.write_y(answer, 'answer.csv')
