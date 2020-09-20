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
import pandas as pd
import xgboost as xgb

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

    # 将未知的数据使用中间值代替
    print(train_data['notRepairedDamage'].value_counts())
    train_data['notRepairedDamage'].replace('-', '0.5', inplace=True)
    train_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype(float)

    print(test_data['notRepairedDamage'].value_counts())
    test_data['notRepairedDamage'].replace('-', '0.5', inplace=True)
    test_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype(float)

    train_data_X = train_data.drop(['price'], axis=1)
    train_data_y = train_data[['price']]

    # 2. 数据概览
    print(f'训练集数据规模: {train_data.shape}\n'
          f'训练集数据描述: {train_data.describe()}')
    print(f'测试集数据规模: {test_data.shape}\n'
          f'测试集数据描述: {test_data.describe()}')

    # 使用 pandas_profiling 进行输出
    # pfr = pandas_profiling.ProfileReport(train_data_X)
    # pfr.to_file('./data/result.html')

    # 3. 数据缺失值处理
    # 3.1 统计缺失值数量
    train_missing = train_data_X.copy().isna()
    test_missing = test_data.copy().isna()

    print(f'训练集缺失值数量: {train_missing.sum()}')
    print(f'测试集缺失值数量: {test_missing.sum()}')

    # 3.2 绘制缺失值分布
    # plot_data(train_missing.sum(), 'bar')
    # plot_data(test_missing.sum(), 'bar')

    # msno.matrix(test_data.sample(250))
    # plt.show()

    # 3.3 缺失值填充
    # 均值填充
    for idx in train_missing.sum()[train_missing.sum() > 0].index:
        train_data_X[idx].fillna(train_data_X[idx].mean(), inplace=True)

    for idx in test_missing.sum()[test_missing.sum() > 0].index:
        test_data.fillna(test_data[idx].mean(), inplace=True)

    # 4. 使用 XGBoost 进行预测
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.01, gamma=0, subsample=0.8, colsample_bytree=0.9,
                             max_depth=7)
    model.fit(train_data_X, train_data_y)
    predict_y = model.predict(test_data)

    # 5. 输出结果
    res = pd.DataFrame()
    res['SaleID'] = test_data['SaleID']
    res['price'] = predict_y
    print(res)
    res.to_csv('./data/used_car/used_car_price_res.csv')
