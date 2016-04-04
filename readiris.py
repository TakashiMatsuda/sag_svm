#!/Users/takashi/.pyenv/shims/python

import csv
import sys


def readiris(filename):
    """
    Extract the item specifying the features and the labels
    """
#    d = sparff.loadarff(filename)
    d = [[] for x in range(2)]
    with open(filename, newline='') as f:
        dreader = csv.reader(f)
        try:
            for row in dreader:
                d[0].append([float(x) for x in row[:-1]])
                d[1].append(1 if row[-1] == 'Iris-setosa' else -1)
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
    return (d[0], d[1])


def test_readiris():
    """
        test readiris(fn)
        task to the code: first entry is correct
    """
    data = readiris("./iris/usingdata.csv")
    assert data[1][0] == 1 and data[1][99] == -1
    assert data[0][82][0] == 5.8
