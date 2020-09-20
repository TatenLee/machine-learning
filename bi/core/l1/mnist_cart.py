# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
利用CART对mnist进行预测
"""

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.title('Handwritten Digits')
plt.imshow(digits.images[0])
plt.show()

# 分隔数据及，比例为7.5:2.5
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# 创建CART分类器
cart = DecisionTreeClassifier()
cart.fit(train_ss_x, train_y)
predict_y = cart.predict(test_ss_x)
print('CART准确率: %0.4lf' % accuracy_score(predict_y, test_y))
