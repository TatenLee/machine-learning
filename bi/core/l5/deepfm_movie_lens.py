# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用 deepFM 对 movie_lens 数据集进行预测
"""

import pandas as pd
from deepctr.inputs import SparseFeat, get_feature_names
from deepctr.models import DeepFM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 加载数据
data = pd.read_csv('data/ratings.csv')
sparse_features = ["movieId", "userId"]
target = ['rating']
# print(train_df.head())
# print(target_df.head())

# 2. 对特征标签进行编码
for feature in sparse_features:
    label = LabelEncoder()
    data[feature] = label.fit_transform(data[feature])

# 3. 计算每个特征中的不同特征值的个数
fix_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
# print(fix_feature_columns)

linear_feature_columns = fix_feature_columns
dnn_feature_columns = fix_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 4. 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

# 5. 使用DeepFM进行训练
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile('adam', 'mse', metrics=['mse'])
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True,
                    validation_split=0.2)

# 6. 使用DeepFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)

# 7. 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print('test RMSE', rmse)
