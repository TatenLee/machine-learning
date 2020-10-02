# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 数据可视化
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.font_manager import FontProperties


def scatter():
    """
    散点图
    :return:
    """
    # 数据准备
    N = 500
    x = np.random.randn(N)
    y = np.random.randn(N)
    # 用 Matplotlib 画散点图
    plt.scatter(x, y, marker='.')
    plt.show()
    # 用 Seaborn 画散点图
    df = pd.DataFrame({'x': x, 'y': y})
    sb.jointplot(x='x', y='y', data=df)
    plt.show()


def line_chart():
    """
    折线图
    :return:
    """
    # 数据准备
    x = [1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910]
    y = [265, 323, 136, 220, 305, 350, 419, 450, 560, 720, 830]
    # 使用Matplotlib画折线图
    plt.plot(x, y)
    plt.show()
    # 使用Seaborn画折线图
    df = pd.DataFrame({'x': x, 'y': y})
    sb.lineplot(x="x", y="y", data=df)
    plt.show()


def bar_chart():
    """
    条形图
    :return:
    """
    # 数据准备
    x = ['c1', 'c2', 'c3', 'c4']
    y = [15, 18, 5, 26]
    # 用 Matplotlib 画条形图
    plt.bar(x, y)
    plt.show()
    # 用 Seaborn 画条形图
    sb.barplot(x, y)
    plt.show()


def box_chart():
    """
    箱线图
    :return:
    """
    # 数据准备
    # 生成0-1之间的20*4维度数据
    data = np.random.normal(size=(10, 4))
    labels = ['A', 'B', 'C', 'D']
    # 用Matplotlib画箱线图
    plt.boxplot(data, labels=labels)
    plt.show()
    # 用Seaborn画箱线图
    df = pd.DataFrame(data, columns=labels)
    sb.boxplot(data=df)
    plt.show()


def pie_chart():
    """
    饼图
    :return:
    """
    # 数据准备
    nums = [25, 33, 37]
    # 射手adc：法师apc：坦克tk
    labels = ['ADC', 'APC', 'TK']
    # 用Matplotlib画饼图
    plt.pie(x=nums, labels=labels)
    plt.show()

    # 数据准备
    data = {'ADC': 25, 'APC': 33, 'TK': 37}
    data = pd.Series(data)
    data.plot(kind="pie", label='heros')
    plt.show()


def thermodynamic():
    """
    热力图
    :return:
    """
    # # 数据准备
    # np.random.seed(33)
    # data = np.random.rand(3, 3)
    # heatmap = sb.heatmap(data)
    # plt.show()

    if os.path.exists("./data/flights.csv"):
        flights = pd.read_csv("./data/flights.csv")
    else:
        flights = sb.load_dataset('flights')
        flights.to_csv("./data/flights.csv")
    flights = flights.pivot('month', 'year', 'passengers')  # pivot函数重要
    sb.heatmap(flights)  # 注意这里是直接传入数据集即可，不需要再单独传入x和y了
    sb.heatmap(flights, linewidth=.5, annot=True, fmt='d')
    plt.show()


def spider_chart():
    """
    蜘蛛图
    :return:
    """
    # 数据准备
    labels = np.array([u"推进", "KDA", u"生存", u"团战", u"发育", u"输出"])
    stats = [76, 58, 67, 97, 86, 58]
    # 画图数据准备，角度、状态值
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    # 用Matplotlib画蜘蛛图
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    ax.set_thetagrids(angles * 180 / np.pi, labels, FontProperties=font)
    plt.show()


def joint_plot():
    """
    二元变量分布图
    :return:
    """
    # 数据准备
    flights = sb.load_dataset("flights")
    # 用Seaborn画二元变量分布图（散点图，核密度图，Hexbin图）
    sb.jointplot(x="year", y="passengers", data=flights, kind='scatter')
    sb.jointplot(x="year", y="passengers", data=flights, kind='kde')
    sb.jointplot(x="year", y="passengers", data=flights, kind='hex')
    plt.show()


def pair_plot():
    """
    成对关系图
    :return:
    """
    # 数据准备
    flights = sb.load_dataset('flights')
    # 用Seaborn画成对关系
    sb.pairplot(flights)
    plt.show()


if __name__ == '__main__':
    thermodynamic()
