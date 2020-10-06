# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 清洗泰坦尼克数据
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def clean():
    # 加载数据
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    print(train_data.info())
    print(train_data.describe())

    # 使用平均年龄来填充年龄中的nan值
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

    # 使用票价的均值填充票价中的nan值
    train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

    # 使用登录最多的港口来填充登录港口的nan值
    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_X = train_data[features]
    train_y = train_data['Survived']
    test_X = test_data[features]

    dvec = DictVectorizer(sparse=False)
    train_X = dvec.fit_transform(train_X.to_dict(orient='record'))
    test_X = dvec.fit_transform(test_X.to_dict(orient='record'))

    return train_X, train_y, test_X
