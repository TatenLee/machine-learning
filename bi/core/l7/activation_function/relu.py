# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    """
    relu 函数

    :param x:
    :return:
    """
    y = np.where(x < 0, 0, x)
    return y


def plot_relu():
    """
    绘制 relu 图

    :return:
    """
    # 设置参数x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)
    y = relu(x)
    plt.plot(x, y)
    plt.show()


plot_relu()
