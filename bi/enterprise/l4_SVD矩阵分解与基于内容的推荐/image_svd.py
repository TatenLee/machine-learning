# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 取前k个特征，对图像进行还原
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import svd


def get_image_feature(s, k):
    """
    获取图像特征
    对于 S，只保留前 K 个特征值

    :param s:
    :param k:
    :return:
    """
    s_tmp = np.zeros(s.shape[0])
    s_tmp[0:k] = s[0:k]
    s = s_tmp * np.identity(s.shape[0])
    # 用新的 s_tmp，以及 p, q 重构 A
    tmp = np.dot(p, s)
    tmp = np.dot(tmp, q)
    plt.imshow(tmp, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    print(A - tmp)


if __name__ == '__main__':
    image = Image.open('./data/256.bmp')
    A = np.array(image)
    # 显示原图像
    plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    # 对图像矩阵 A 进行奇异值分解，得到 p, s, q
    p, s, q = svd(A, full_matrices=False)
    # 取前 k 个特征，对图像进行还原
    get_image_feature(s, 5)
    get_image_feature(s, 50)
    get_image_feature(s, 500)
