# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用 ARMA 预测股票走势
"""

import calendar
from datetime import timedelta
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA


def plot_trend(data):
    """
    按照天、月、季度、年来显示股票走势

    :param data:
    :return:
    """
    # 按照月、季度、年来统计
    df_day = data
    df_month = data.resample('M').mean()
    df_quarter = data.resample('Q-DEC').mean()
    df_year = data.resample('A-DEC').mean()

    plt.figure(figsize=[15, 7])
    plt.suptitle('股票', fontsize=20)
    plt.subplot(221)
    plt.plot(df_day['Price'], '-', label='days')
    plt.legend()
    plt.subplot(222)
    plt.plot(df_month['Price'], '-', label='months')
    plt.legend()
    plt.subplot(223)
    plt.plot(df_quarter['Price'], '-', label='quarter')
    plt.legend()
    plt.subplot(224)
    plt.plot(df_year['Price'], '-', label='year')
    plt.legend()
    plt.show()


# 1. 加载数据
df = pd.read_csv('data/600585.csv', encoding='GB2312')
df = df[['日期', '收盘价']].rename(columns={'日期': 'Date', '收盘价': 'Price'}, copy=True)
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
df.drop(columns=['Date'], inplace=True)

# 2. 数据探索
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plot_trend(df)
month_df = df.resample('M').mean()

# 3. 建立模型
ps = range(0, 3)
qs = range(0, 3)
params = product(ps, qs)
param_list = list(params)
print(param_list)

# 4. 寻找最优 ARMA 模型参数
res = []
best_model = None
best_aic = float('inf')  # 正无穷
for param in param_list:
    try:
        model = ARMA(month_df['Price'], order=(param[0], param[1])).fit()
    except ValueError:
        print(f'参数错误: {param}')
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    res.append([param, model.aic])

# 5. 输出最优模型
print(f'最优模型: {best_model.summary()}')

# 6. 设置 future_month，需要预测的时间 date_list
month_price_df = month_df[['Price']]
future_month_num = 3
last_month = pd.to_datetime(month_price_df.index[len(month_price_df) - 1])

date_list = []
for i in range(future_month_num):
    # 计算下个月有多少天
    year = last_month.year
    month = last_month.month
    if month == 12:
        month = 1
        year = year + 1
    else:
        month = month + 1
    next_mont_days = calendar.monthrange(year, month)[1]
    last_month = last_month + timedelta(days=next_mont_days)
    date_list.append(last_month)
print(f'date_list: {date_list}')

# 添加未来要预测的3个月
future_month = pd.DataFrame(index=date_list, columns=month_df.columns)
month_price_df = pd.concat([month_price_df, future_month])
month_price_df['Forecast'] = best_model.predict(start=0, end=len(month_price_df))
month_price_df.loc[:1, 'Forecast'] = np.NaN
print(month_price_df)

plt.figure(figsize=(30, 7))
month_price_df['Price'].plot(label='Actual')
month_price_df['Forecast'].plot(label='Forecast')
plt.legend()
plt.title('股票')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
