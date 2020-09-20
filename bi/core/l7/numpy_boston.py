# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用 numpy 进行波士顿房价预测
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

# 1. 加载数据
data = load_boston()
X = data['data']
y = data['target']
print(y.shape)
y = y.reshape(y.shape[0], 1)
print(y)
