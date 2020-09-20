# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description:
1）计算选定股票，上证综指的每日收益率
2）计算这两个每日收益率之间的线性回归方程
"""
import datetime
import os

import pandas as pd
import statsmodels.api as sm  # 回归分析
from pandas_datareader.data import DataReader
import matplotlib.pyplot as plt

days_delta = 30
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=days_delta)
spec_stock_code_list = ['000001.SS', '000063.SZ']


def load_data():
    """
    加载数据集

    :return:
    """
    data_info_dict = {}
    for stock_code in spec_stock_code_list:
        if not os.path.exists(f'{stock_code.split(".")[0]}.csv'):
            # 获取列表中的股票信息
            data = DataReader(stock_code, 'yahoo', start=start_date, end=end_date)
            # 保存为本地的csv文件
            data.to_csv(f'{stock_code.split(".")[0]}.csv')

        data_df = pd.read_csv(f'{stock_code.split(".")[0]}.csv')
        data_info_dict[stock_code.split('.')[0]] = data_df
    return data_info_dict


if __name__ == '__main__':
    stock_info_dict = load_data()
    stock = pd.DataFrame()

    for stock_code in stock_info_dict.keys():
        if stock.empty:
            stock_df = stock_info_dict[stock_code]
            stock_df.columns = [f'{column}_{stock_code}' for column in stock_df.columns]
            stock = stock_df
        else:
            stock_df = stock_info_dict[stock_code]
            stock_df.columns = [f'{column}_{stock_code}' for column in stock_df.columns]
            stock = pd.merge(stock, stock_df, left_index=True, right_index=True)

    stock = stock[['Close_000001', 'Close_000063']]
    stock.columns = ['上证综指', '中兴通讯']

    # 计算每日收益率
    daily_return = (stock.diff() / stock.shift(periods=1)).dropna()
    print(daily_return)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    daily_return['上证综指'].plot(ax=ax[0])
    ax[0].set_title('上证综指')
    daily_return['中兴通讯'].plot(ax=ax[1])
    ax[1].set_title('中兴通讯')

    plt.show()

    # 加入截距项
    daily_return["intercept"] = 1.0
    model = sm.OLS(daily_return['中兴通讯'], daily_return[['上证综指', 'intercept']])
    results = model.fit()
    print(results.summary())

    # 得到线性回归方程
    k = results.params['上证综指']
    b = results.params['intercept']
    print(f'方程为: y = {k}x{b}')
