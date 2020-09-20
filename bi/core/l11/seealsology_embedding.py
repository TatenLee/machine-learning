# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: seealsology是个针对Wikipidea页面的语义分析工具，可以找到与指定页面相关的Wikipidea
seealsology-data.tsv 文件存储了Wikipidea页面的关系（Source, Target, Depth）
使用Graph Embedding对节点（Wikipidea）进行Embedding（DeepWalk或Node2Vec模型）
对Embedding进行可视化（使用PCA呈现在二维平面上）
找到和critical illness insurance相关的页面
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from node2vec import Node2Vec
from sklearn.decomposition import PCA

# 显示所有列
pd.set_option('display.max_columns', None)


def plot_nodes(word_list):
    """
    在二维空间中绘制所选节点的向量

    :param word_list:
    :return:
    """
    # 每个节点的 embedding 为 100 维
    X = []
    for word in word_list:
        X.append(embedding[word])
    # 将 100 维向量减少到 2 维
    pca = PCA(n_components=2)
    res = pca.fit_transform(X)
    # 绘制节点向量
    plt.figure(figsize=(12, 9))
    # 创建一个散点图的投影
    plt.scatter(res[:, 0], res[:, 1])
    for i, word in enumerate(list(word_list)):
        plt.annotate(word, xy=(res[i, 0], res[i, 1]))
    plt.show()


if __name__ == '__main__':
    # 1. 加载数据
    df = pd.read_csv('data/seealsology-data.tsv', sep='\t')
    G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr=True, create_using=nx.Graph)
    print(type(G))
    print(len(G))

    # 2. 初始化 Node2Vec 模型
    model = Node2Vec(G, walk_length=10, num_walks=5, p=0.25, q=4, workers=1)

    # 3. 模型训练
    result = model.fit()

    # 4. 获取节点的 embedding
    print(result.wv.most_similar('critical illness insurance'))
    embedding = result.wv
    print(embedding)

    # 5. 绘制图像
    plot_nodes(result.wv.vocab)
