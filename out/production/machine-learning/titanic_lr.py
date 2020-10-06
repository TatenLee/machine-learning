# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import sys

from sklearn.linear_model import LogisticRegression

sys.path.append('./')

from titanic_clean import clean

if __name__ == '__main__':
    train_X, train_y, test_X = clean()

    # 选择模型
    model_lr = LogisticRegression(max_iter=500)

    # 模型训练
    model_lr.fit(train_X, train_y)

    # 模型预测
    y_pred = model_lr.predict(test_X)

    # 评分（基于训练集）
    score = model_lr.score(train_X, train_y)
    print(score)
