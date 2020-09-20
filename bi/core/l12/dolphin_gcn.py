# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: Dolphin数据集是 D.Lusseau 等人使用长达 7 年的时间
观察新西兰 Doubtful Sound海峡 62 只海豚群体的交流情况而得到的海豚社会关系网络。
这个网络具有 62 个节点，159 条边。节点表示海豚，边表示海豚间的频繁接触。
1. 对Dolphin 关系进行Graph Embedding，可以使用DeepWalk, Node2Vec或GCN
2. 对Embedding进行可视化（使用PCA呈现在二维平面上）
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA


def plot_data(graph):
    """
    绘制网络图

    :param graph:
    :return:
    """
    plt.figure()
    pos = nx.spring_layout(graph)
    edges = graph.edges()
    nodes = graph.nodes()
    nx.draw_networkx(graph, pos, edges)
    nx.draw_networkx_nodes(graph, pos, nodes, node_size=300, node_color='r', alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edges, alpha=0.4)
    plt.show()


def relu(x):
    """
    激活函数
    当 x<0 时，结果=0
    当 x>=0 时，结果=x
    :param x:
    :return:
    """
    return (abs(x) + x) / 2


def plot_node(nodes, output, title):
    """
    绘制output，节点 GCN embedding 可视化

    :param nodes:
    :param output:
    :param title:
    :return:
    """
    for i in range(len(nodes)):
        plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], label=str(i), alpha=0.5, s=250)
        plt.text(np.array(output)[i, 0], np.array(output)[i, 1], i, horizontalalignment='center',
                 verticalalignment='center', fontdict={'color': 'black'})
    plt.title(title)
    plt.show()


def gcn_layer(A_hat, D_hat, X, W):
    """
    叠加 GCN 层，使用单位矩阵作为特征表征，即每个节点被表示为一个 one-hot 编码的类别变量

    :param A_hat:
    :param D_hat:
    :param X:
    :param W:
    :return:
    """
    return relu(D_hat ** -1 * A_hat * X * W)


def run():
    # 1. 加载数据
    G = nx.read_gml('data/dolphins.gml')

    # 2. 数据可视化
    # plot_data(G)

    # 3. 构建 GCN， 计算 A_hat 和 D_hat 矩阵
    dolphin_list = sorted(list(G.nodes))

    A = nx.to_numpy_matrix(G, dolphin_list)
    # 生成对角矩阵
    I = np.eye(G.number_of_nodes())

    # 生成 A_hat
    A_hat = A + I
    print(A_hat)

    # 生成 D_hat
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    print(D_hat)
    # 得到对角线上的元素
    D_hat = np.matrix(np.diag(D_hat))
    print(D_hat)

    # 4. 初始化权重，normal 正态分布，loc均值，scale标准差
    W_1 = np.random.normal(loc=0, scale=1, size=(G.number_of_nodes(), 4))
    W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

    # 5. 叠加 GCN 层
    H_1 = gcn_layer(A_hat, D_hat, I, W_1)
    H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
    output = H_2
    print(f'叠加 GCN 层后的 output:\n{output}')

    # 6. 提取特征表征
    feature_representations = {}
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        feature_representations[nodes[i]] = np.array(output)[i]
    print(f'feature_representations=\n{feature_representations}')

    # 7. GCN embedding 使用 PCA 可视化
    print(output)
    pca = PCA(n_components=2)
    res = pca.fit_transform(output)
    print(res)
    plot_node(nodes, res, title='Graph Embedding')


if __name__ == '__main__':
    run()
