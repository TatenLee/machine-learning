# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 基于能力描述的薪资预测
数据集：抓取了4512个职位的能力描述，薪资
Step1，数据加载
Step2，可视化，使用Networkx
Step3，提取文本特征 TFIDF
Step4，回归分析，使用KNN回归，朴素贝叶斯回归，训练能力和薪资匹配模型
Step5，基于指定的能力关键词，预测薪资
"""
import random
import re

import jieba
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor

# 显示所有列
pd.set_option('display.max_columns', None)

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


def handle_job_string(row):
    """
    处理每一行数据

    :param row:
    :return:
    """
    job_string = ''
    for idx, element in enumerate(row.split('\n')):
        if len(element.split()) == 2:
            idx, value = element.split()
            if idx == 0:
                continue
            job_string += value
    return job_string


def predict_by_label(test_string, vectorizer, model):
    """
    通过能力标签预测薪资

    :param test_string:
    :param vectorizer:
    :param model:
    :return:
    """
    test_words = list(jieba.cut(test_string))
    test_vec = vectorizer.transform(test_words)
    predict_value = model.predict(test_vec)
    return predict_value[0]


def main():
    # 1. 数据加载
    data = pd.read_excel('./data/jobs_4k.xls')
    print(data.head())

    # 2. 观察职位和技能之间的关联度，并可视化
    position_names = data['positionName'].tolist()
    skill_labels = data['skillLabels'].tolist()

    # 构建职位技能字典
    position_to_skill_dict = dict()
    for p, s in zip(position_names, skill_labels):
        if position_to_skill_dict.get(p) is None:
            position_to_skill_dict[p] = eval(s)
        else:
            position_to_skill_dict[p] += eval(s)
    print(position_to_skill_dict)

    # 随机选择k个工作岗位
    sample_nodes = random.sample(position_names, k=5)
    print(f'随机选择k个工作岗位: \n{sample_nodes}')

    # 将职位信息和能力描述结合到一起
    sample_nodes_connections = sample_nodes
    for p, s in position_to_skill_dict.items():
        if p in sample_nodes:
            sample_nodes_connections += s
    print(f'将职位和能力放在一起: \n{sample_nodes_connections}')

    # 绘制图像
    G = nx.Graph(position_to_skill_dict)

    # 抽取原始G中的节点作为子图
    sample_graph = G.subgraph(sample_nodes_connections)
    plt.figure(figsize=(16, 8))
    pos = nx.spring_layout(sample_graph, k=1)
    nx.draw(sample_graph, pos, with_labels=True)
    # plt.show()

    # 使用PageRank计算节点（技能）影响力
    pr = nx.pagerank(G, alpha=0.9)
    ranked_position_skill = sorted([(name, value) for name, value in pr.items()], key=lambda _: _[1], reverse=True)
    print(f'排序后的职位和技能: \n{ranked_position_skill}')

    # 3. 构造特征，使用TF-IDF提取文本特征
    data_X = data.drop(['salary'], axis=1).copy()
    targets = data['salary'].tolist()

    # 将所有的特征放在一起
    data_X['merged'] = data_X.apply(lambda _: ''.join(str(_)), axis=1)

    # 分词处理
    cut_X = list()
    for idx, row in enumerate(data_X['merged']):
        job_string = handle_job_string(row)
        cut_X.append(' '.join(list(jieba.cut(''.join(re.findall('\\w+', job_string))))))
    print(f'分词处理后的数据: \n{cut_X[0]}')

    # 使用 TF-IDF 提取文本特征
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cut_X)
    print(f'提取文本特征后的数据: \n{X[0]}')

    # 求平均值 薪资 10k - 15k => 12.5k
    target_numerical = [np.mean(list(map(float, re.findall('\\d+', target)))) for target in targets]
    Y = target_numerical

    # 4. 回归分析，使用KNN回归，训练能力和薪资匹配模型
    model = KNeighborsRegressor(n_neighbors=2)
    model.fit(X, Y)
    print(f'KNN模型评分为: {model.score}')

    # 5. 基于指定的能力关键词，预测薪资
    test_string_list = [
        '测试 北京 3年 专科',
        '测试 北京 4年 专科',
        '算法 北京 4年 本科',
        'UI 北京 4年 本科',
        '广州Java本科3年掌握大数据',
        '沈阳Java硕士3年掌握大数据',
        '沈阳Java本科3年掌握大数据',
        '北京算法硕士3年掌握图像识别'
    ]

    for test_string in test_string_list:
        print(f'职位和能力信息: {test_string}, 预测的薪资为: {predict_by_label(test_string, vectorizer, model)}')


if __name__ == '__main__':
    main()
