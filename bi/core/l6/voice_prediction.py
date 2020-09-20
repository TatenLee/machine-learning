# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 男女声音识别
3168个录制的声音样本（来自男性和女性演讲者），采集的频率范围是0hz-280hz，已经对数据进行了预处理
一共有21个属性值，请判断该声音是男还是女？
使用Accuracy作为评价标准
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

params = {
    'max_depth': 15,
    'alpha': 0.6
}

data = pd.read_csv('data/voice.csv')
train = data.drop(columns=['label'], axis=1)
test = data[['label']].copy()
test['label'] = test['label'].map(lambda _: 1 if _ == 'male' else 0)

X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

bst = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dtest, 'test')], num_boost_round=10000,
                early_stopping_rounds=200, verbose_eval=25)
pred = bst.predict(dtest)
pred = [round(value) for value in pred]
accuracy_score = accuracy_score(y_test, pred)
print(accuracy_score)
