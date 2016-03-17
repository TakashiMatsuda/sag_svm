#!/Users/takashi/.pyenv/shims/python

from scipy.io import arff as sparff
import numpy as np


def read_data(filename):
    """
    Extract the item specifying the features and the labels
    """
    d = sparff.loadarff(filename)
    data = d[0]
    return datacleaning(data)


def datacleaning(data):
    n_data = np.zeros((len(data), len(data[0])))
    misslist = []
    for i, v_x in enumerate(data):
        for j, vv_x in enumerate(v_x):
            if vv_x == bytes(b'?'):
                # REMOVE this entry
                misslist.append(i)
                break
            # 以下列挙
            elif j in {2, 3, 4}:
                n_data[i][j] = float(vv_x)
                continue
            elif j in {5, 6}:
                n_data[i][j] = (1) if (vv_x == bytes(b'normal')) else (0)
                continue
            elif j in {7, 8}:
                n_data[i][j] = (1) if (vv_x == bytes(b'present')) else (0)
                continue
            elif j in {18, 19, 20, 22, 23}:
                n_data[i][j] = (1) if (vv_x == bytes(b'yes')) else (0)
                continue
            elif j in {21}:
                n_data[i][j] = (1) if (vv_x == bytes(b'good')) else (0)
                continue
            elif j in {24}:
                n_data[i][j] = (1) if (vv_x == bytes(b'ckd')) else (0)
                continue
            else:
                n_data[i][j] = vv_x
                continue

    return n_data


def test_readdata():
    fn = "./chronic_kidney_disease/Chronic_Kidney_Disease_full.arff"
    d = read_data(fn)
    print(d)
    assert d[0][0] == 48
