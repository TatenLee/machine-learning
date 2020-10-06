# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 矩阵奇异值分解
"""

import numpy as np
from scipy.linalg import svd


def matrix_decomposition():
    """
    矩阵分解

    :return:
    """
    A = np.array([
        [1, 2],
        [1, 1],
        [0, 0]
    ])

    tmp1 = np.dot(A, A.T)
    tmp2 = np.dot(A.T, A)
    print(f'tmp1: {tmp1}\n')
    print(f'tmp2: {tmp2}\n')

    lambda1, U1 = np.linalg.eig(tmp1)
    print(f'lambda1: {lambda1}\n')
    print(f'U1: {U1}\n')

    lambda2, U2 = np.linalg.eig(tmp2)
    print(f'lambda2: {lambda2}\n')
    print(f'U2: {U2}\n')


def svd_decomposition():
    """
    SVD 分解

    :return:
    """
    A = np.array([
        [1, 2],
        [1, 1],
        [0, 0]
    ])
    p, s, q = svd(A, full_matrices=False)
    print('P=', p)
    print('S=', s)
    print('Q=', q)


if __name__ == '__main__':
    matrix_decomposition()
    svd_decomposition()
