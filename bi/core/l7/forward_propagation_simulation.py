# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用 numpy 模拟前向传播
"""

import numpy as np


def init_network():
    """
    初始化网络（初始权重和偏置）

    :return:
    """
    network = dict()
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def sigmoid(x):
    """
    sigmoid 函数

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    """
    恒等函数，作为输出层的激活函数

    :param x:
    :return:
    """
    return x


def forward(network, x):
    """
    前向传播

    :param network:
    :param x:
    :return:
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


if __name__ == '__main__':
    # 初始化网络
    network = init_network()
    # 设置输入值
    x = np.array([1.0, 0.5])
    # 前向传播
    y = forward(network, x)
    # 打印结果
    print(y)
