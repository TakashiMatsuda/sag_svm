#!/Users/takashi/.pyenv/shims/python


def read_x(filepath):
    """
    Read data in 'filepath'
    """
    dim_strain = 0
    x = [[0 for j in range(dim_strain)] for i in range(dim_strain)]
    return x


def test_read_x():
    file1 = "testfile.csv"
    assert read_x(file1)
