# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 利用 LSTM 预测股票价格
"""
import os
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model, Sequential

TIME_STEPS_IN = 3
TIME_STEPS_OUT = 3
EPOCHS = 300
BATCH_SIZE = 100


def series_to_supervised(series, n_in=1, n_out=1, drop_nan=True):
    """
    将时间序列数据转换为适用于监督学习的数据
    给定输入序列和输出序列的长度

    :param series: 观察序列
    :param n_in: 观测数据 input(X) 的步长，范围是 [1, len(data)]，默认为1
    :param n_out: 观测数据 output(y) 的步长，范围为 [0, len(data)-1]，默认为1
    :param drop_nan: 是否删除 NaN 行，默认为 True
    :return: 适用于监督学习的数据集
    """
    n_vars = 1 if type(series) is list else series.shape[1]
    df = pd.DataFrame(series)
    cols, names = list(), list()
    # 输入序列 (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j + 1}(t-{i})' for j in range(n_vars)]

    # 预测序列 (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j + 1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j + 1}(t+{i})' for j in range(n_vars)]

    # 拼接到一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if drop_nan:
        agg.dropna(inplace=True)
    return agg


def get_train_set(dataset, time_steps_in=1, time_steps_out=1):
    """
    将数据转换为可用于监督学习的数据

    :param dataset:
    :param time_steps_in:
    :param time_steps_out:
    :return:
    """
    train_dataset = np.array(dataset)
    print(train_dataset)
    reframed_train_dataset = np.array(series_to_supervised(train_dataset, time_steps_in, time_steps_out).values)
    print(reframed_train_dataset)
    train_x, train_y = reframed_train_dataset[:, :-time_steps_out], reframed_train_dataset[:, -time_steps_out:]
    # 将数据集重构为符合 LSTM 要求的数据格式，即：[样本数，时间步，特征]
    train_x = train_x.reshape(train_x.shape[0], time_steps_in, 1)
    return train_x, train_y


def plot_img(source_dataset, train_predict):
    """
    绘制图像

    :param source_dataset:
    :param train_predict:
    :return:
    """
    plt.figure(figsize=(24, 8))
    # 原始数据蓝色
    plt.plot(source_dataset[:, -1], c='b', label='actual')
    # 训练数据绿色
    plt.plot([_ for _ in train_predict], c='g', label='predict')
    plt.legend()
    plt.show()


def lstm_model(source_dataset, train_data, label_data, epochs, batch_size, time_steps_out):
    if os.path.exists('data/model.h5'):
        model = load_model('data/model.h5')
    else:
        model = Sequential()
        # 第一层，隐藏层神经元节点个数为128，返回整个序列
        model.add(
            LSTM(128, return_sequences=True, activation='relu', input_shape=(train_data.shape[1], train_data.shape[2]))
        )
        # 第二层，隐藏层神经元节点个数128，只返回序列最后一个输出
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.5))
        # 第三层，因为是回归问题所以使用linear
        model.add(Dense(time_steps_out, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.save('./data/model.h5', overwrite=True)

    # LSTM 训练
    # verbose=2 为每个epoch输出一行记录；
    # verbose=1 为输出进度条记录；
    # verbose=0 不在标准输出流输出日志信息
    res = model.fit(train_data, label_data, batch_size, epochs, verbose=2, shuffle=False)

    # 模型预测
    train_predict = model.predict(train_data)
    train_predict_list = list(chain(*train_predict))

    plt.plot(res.history['loss'], label='train')
    plt.show()
    print(model.summary())
    plot_img(source_dataset, train_predict)


if __name__ == '__main__':
    data = pd.read_csv('data/shanghai_index_1990_12_19_to_2020_03_12.csv', encoding='GB2312')
    data.rename(columns={'日期': 'Date', '收盘价': 'Price'}, inplace=True)
    data_set = data[['Price']].values.astype('float64')
    print(data_set)
    # 转换为可用于监督学习的数据
    train_X, label_y = get_train_set(data_set, TIME_STEPS_IN, TIME_STEPS_OUT)
    # 使用 LSTM 进行训练、预测
    lstm_model(data_set, train_X, label_y, EPOCHS, BATCH_SIZE, TIME_STEPS_OUT)
