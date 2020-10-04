# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import sys

from sklearn import ensemble
from sklearn.metrics import mean_squared_error

sys.path.append('./')

from titanic_clean import clean

if __name__ == '__main__':
    train_X, train_y, test_X = clean()

    # 模型选择
    model_gbdt = ensemble.GradientBoostingClassifier(n_estimators=500, max_depth=5, learning_rate=0.01)

    # 模型训练
    model_gbdt.fit(train_X, train_y)

    # 模型预测
    y_pred = model_gbdt.predict(test_X)

    # 模型评分
    train_y_pred = model_gbdt.predict(train_X)
    mse = mean_squared_error(train_y_pred, train_y)
    print(mse)
