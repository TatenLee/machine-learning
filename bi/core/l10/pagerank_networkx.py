# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用 networkx 计算节点的 pagerank
"""

import networkx as nx
import matplotlib.pylab as plt

# 创建有向图
G = nx.DiGraph()

# 设置有向图的边集合
edges = [('A', 'B'), ('A', 'D'), ('A', 'E'), ('A', 'F'), ('B', 'C'), ('C', 'E'), ('D', 'A'), ('D', 'C'), ('D', 'E'),
         ('E', 'B'), ('E', 'C'), ('F', 'D')]

# 在有向图G中添加边集合
for edge in edges:
    G.add_edge(edge[0], edge[1])

# 有向图可视化
layout = nx.spring_layout(G)
nx.draw(G, pos=layout, with_labels=True, hold=False)
plt.show()

# 计算简化模型的 PR 值
pr = nx.pagerank(G, alpha=1)
print(f'简化模型的PR值: {pr}')

# 计算随机模型的 PR 值
pr = nx.pagerank(G, alpha=0.8)
print(f'随机模型的PR值: {pr}')
