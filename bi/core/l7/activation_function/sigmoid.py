# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 绘制 sigmoid 函数
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    sigmoid 函数

    :param x:
    :return:
    """
    y = 1 / (1 + np.exp(-x))
    return y


def derivative_sigmoid(x):
    """
    sigmoid 函数的导数

    :param x:
    :return:
    """
    y = sigmoid(x)
    dy = y * (1 - y)
    return dy


def plot_sigmoid(is_derivative):
    # 设置参数 x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)

    if is_derivative:
        y = derivative_sigmoid(x)
    else:
        y = sigmoid(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    plot_sigmoid(False)
