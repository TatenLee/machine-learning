# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# 加载数据
file_name = 'data/sample.csv'
df = pd.read_csv(file_name)

# 设置最大显示列数
pd.options.display.max_columns = 100

# 计算 CVR
# 行为种类包括: 1 浏览；2 收藏；3 加购物车；4 购买
count_user = df['behavior_type'].value_counts()
count_all = count_user[1] = count_user[2] + count_user[3] + count_user[4]
count_buy = count_user[4]
cvr = count_buy / count_all
print(f'cvr={cvr * 100:.2f}%')  # 18.40

# 将time字段设置为pandas里面的datetime类型
df['time'] = pd.to_datetime(df['time'])
df.index = df['time']
df.drop(['time'], axis=1, inplace=True)


def plot_diagram(data, kind):
    """
    绘制图表

    :param data:
    :param kind:
    :return:
    """
    data.plot(kind=kind)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def show_count_day(data):
    """
    时间规律统计

    :param data:
    :return:
    """
    count_day = dict()
    # 从 2014-11-18 到 2014-12-18
    current_day = '2014-11-18'
    # 字符串类型转换为时间类型
    current_date = datetime.strptime(current_day, '%Y-%m-%d')
    delta = timedelta(days=1)
    for i in range(30):
        # 时间类型转换为字符串
        current_day = current_date.strftime('%Y-%m-%d')
        print(current_day)
        if count_day.get(current_day):
            count_day[current_day] = count_day[current_day] + data[current_day].shape[0]
        else:
            count_day[current_day] = data[current_day].shape[0]
        current_date = current_date + delta
    print(count_day)

    df_count_day = pd.DataFrame.from_dict(count_day, orient='index', columns=['count'])
    plot_diagram(df_count_day['count'], 'bar')


def show_count_hour(date):
    """
    基于小时的统计分析是怎样的，可以指定某个日期，比如11月18日

    :param date:
    :return:
    """
    count_hour = {}

    for i in range(24):
        time_str = f'{date} {i:02d}'

        # 设置初始值
        count_hour[time_str] = [0, 0, 0, 0]
        tmp = df[time_str]['behavior_type'].value_counts(())
        print(tmp)
        for j in range(len(tmp)):
            count_hour[time_str][tmp.index[j] - 1] = tmp[tmp.index[j]]

    print(count_hour)
    df_count_hour = pd.DataFrame.from_dict(count_hour, orient='index')
    plot_diagram(df_count_hour, 'bar')


# 属于商品自己P的操作次数
df_p = pd.read_csv('data/tianchi_fresh_comp_train_item.csv')

# 合并用户和商品数据
df = pd.merge(df.reset_index(), df_p, on=['item_id']).set_index('time')

# 展示每天的用户操作分布情况
show_count_day(df)

# 展示指定日期每小时用户操作分布情况
show_count_hour('2014-12-12')
