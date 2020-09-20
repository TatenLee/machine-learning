# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 绘制 tanh 函数
"""

import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    """
    tanh 函数

    :param x:
    :return:
    """
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y


def plot_tanh():
    """
    绘制 tanh 函数

    :return:
    """
    # 设置参数x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)
    y = tanh(x)
    plt.plot(x, y)
    plt.show()


plot_tanh()
