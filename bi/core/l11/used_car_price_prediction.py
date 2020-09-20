# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 二手车价格预测
数据集：
used_car_train_20200313.csv
used_car_testA_20200313.csv
数据来自某交易平台的二手车交易记录
1、数据探索EDA（20points）
2、使用缺失值可视化工具或pandas_profiling工具（10points）
ToDo：给你一辆车的各个属性（除了price字段），预测它的价格
"""

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import pandas_profiling
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)


def plot_data(data, plot_type):
    """
    绘制图像

    :param data:
    :param plot_type:
    :return:
    """
    if plot_type == 'bar':
        data.plot.bar()
        plt.show()


if __name__ == '__main__':
    # 1. 数据加载
    train_data = pd.read_csv('data/used_car/used_car_train_20200313.csv', sep=' ')
    test_data = pd.read_csv('data/used_car/used_car_testB_20200421.csv', sep=' ')

    print(train_data.info())
    train_data_X = train_data.drop(['price', 'notRepairedDamage'], axis=1)
    train_data_y = train_data[['price']]

    # 2. 数据概览
    print(f'训练集数据规模: {train_data.shape}\n'
          f'训练集数据描述: {train_data.describe()}')
    print(f'测试集数据规模: {test_data.shape}\n'
          f'测试集数据描述: {test_data.describe()}')

    # 3. 数据缺失值处理
    # 3.1 统计缺失值数量
    train_missing = train_data_X.copy().isna()
    test_missing = test_data.copy().isna()

    print(f'训练集缺失值数量: {train_missing.sum()}')
    print(f'测试集缺失值数量: {test_missing.sum()}')

    # 3.2 绘制缺失值分布
    plot_data(train_missing.sum(), 'bar')
    plot_data(test_missing.sum(), 'bar')

    msno.matrix(test_data.sample(250))
    plt.show()

    # 3.3 缺失值填充
    # 均值填充
    for idx in train_missing.sum()[train_missing.sum() > 0].index:
        train_data_X[idx].fillna(train_data_X[idx].mean(), inplace=True)

    # 4. 使用pandas_profiling进行输出
    pfr = pandas_profiling.ProfileReport(train_data_X)
    pfr.to_file('./data/result.html')

    # 5. 使用线性回归进行预测
    # TODO 这里不太会处理非数值型，知道可以用one-hot，或者labelEncoder，但是不会写代码，只好先drop掉。
    # TODO 但是 drop 掉 notRepairedDamage 这列数据以后还是会报错（ValueError: could not convert string to float: '-'）
    # TODO 希望老师指点
    model = LinearRegression()
    model.fit(train_data_X, train_data_y)
    predict_data = model.predict(test_data)
    print(predict_data)
