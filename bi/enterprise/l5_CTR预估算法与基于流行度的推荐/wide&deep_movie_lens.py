# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用 Wide&Deep 进行预测
"""

import pandas as pd
from deepctr.inputs import SparseFeat, get_feature_names
from deepctr.models import WDL
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('./data/movielens_sample.txt')
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 对特征标签进行编码
    for feature in sparse_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])

    # 计算每个特征中不同特征值的个数
    fix_len_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
    linear_feature_columns = fix_len_feature_columns
    dnn_feature_columns = fix_len_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 将数据集切分成训练集和测试集
    train, test = train_test_split(data, test_size=0.2)
    train_set = {name: train[name].values for name in feature_names}
    test_set = {name: test[name].values for name in feature_names}

    # 使用 WDL 进行训练
    model = WDL(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile('adam', 'mse', metrics=['mse'])
    history = model.fit(train_set, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2)

    # 使用 WDL 进行预测
    pred_ans = model.predict(test_set, batch_size=256)

    # 输出 RMSE 或者 MSE
    mse = round(mean_squared_error(test[target].values, pred_ans), 4)
    rmse = mse ** 0.5
    print(f'test rmse: {rmse}')
