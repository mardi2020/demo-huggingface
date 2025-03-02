# https://github.com/ExcelsiorCJH/DLFromScratch2/blob/master/dataset/spiral.py
import numpy as np


def load_data(seed=1984):
    np.random.seed(seed)
    N, DIM, CLS_NUM = 100, 2, 3

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int64)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = 4.0 * (rate + j) + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t
