import numpy as np

col1, col2 = np.loadtxt('NGC7331.txt', unpack=True)


def sum(x, y):
    return x + y


def test_stum():
    assert sum(col1[0], col2[0]) == 33.474959999999996
