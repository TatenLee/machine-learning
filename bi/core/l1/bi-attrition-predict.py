# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
kaggle 预测员工离职率
https://www.kaggle.com/c/bi-attrition-predict/
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    sample_data_df = pd.read_csv('../../resource/bi-attrition-predict/sample.csv')
    train_data_df = pd.read_csv('../../resource/bi-attrition-predict/train.csv')
    test_data_df = pd.read_csv('../../resource/bi-attrition-predict/test.csv')

    # 1. 划分训练集和测试集
    X, y = train_data_df.drop(['Attrition', 'user_id'], 1), train_data_df['Attrition']

    train_x, train_y, test_x, test_y = train_test_split(X, y, test_size=0.25, random_state=33)

    # 2. 对特征进行归一化
    # ss = preprocessing.Normalizer()
    # train_ss_x = ss.fit_transform(train_x)
    # test_ss_x = ss.transform(test_x)

    # 3. 模型训练
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predict_y = lr.predict(test_x)
    print('LR准确率: %0.4lf' % accuracy_score(predict_y, test_y))
