# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import numpy as np

"""
    A   B   C   D   E   F
A   0   0   0   1/3 0   0 
B   1/4 0   0   0   1/2 0
C   0   1   0   1/3 1/2 0
D   1/4 0   0   0   0   1
E   1/4 0   1   1/3 0   0
F   1/4 0   0   0   0   0
"""

# 迭代次数
ITER_TIMES = 100

# 构建状态转移矩阵
TRANSFER_MATRIX = np.array([
    [0, 0, 0, 1 / 3, 0, 0],
    [1 / 4, 0, 0, 0, 1 / 2, 0],
    [0, 1, 0, 1 / 3, 1 / 2, 0],
    [1 / 4, 0, 0, 0, 0, 1],
    [1 / 4, 0, 1, 1 / 3, 0, 0],
    [1 / 4, 0, 0, 0, 0, 0]
])

# 构建权重矩阵
WEIGHT_MATRIX = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])


def simple_model(matrix, weight, iter_times):
    """
    简化模型

    :param matrix:
    :param weight:
    :param iter_times:
    :return:
    """
    for i in range(iter_times):
        weight = np.dot(matrix, weight)
        print(weight)


def random_model(matrix, weight, rank, iter_times, damping_factor=0.85):
    """
    随机模型

    :param matrix:
    :param weight:
    :param rank: 秩
    :param iter_times: 迭代次数
    :param damping_factor: 阻尼因子
    :return:
    """
    for i in range(iter_times):
        weight = (1 - damping_factor) / rank + damping_factor * np.dot(matrix, weight)
        print(weight)


if __name__ == '__main__':
    # 使用简单模型
    simple_model(TRANSFER_MATRIX, WEIGHT_MATRIX, ITER_TIMES)
    print('-' * 100)
    # 使用随机模型
    random_model(TRANSFER_MATRIX, WEIGHT_MATRIX, 6, ITER_TIMES)
    print('*' * 100)
    random_model(TRANSFER_MATRIX, WEIGHT_MATRIX, 6, ITER_TIMES, damping_factor=0.8)
