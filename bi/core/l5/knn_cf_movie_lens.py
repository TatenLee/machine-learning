# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用邻域的协同过滤对movie lens进行预测，并采用K折交叉验证
"""

from surprise import KNNWithZScore, Reader, Dataset
from surprise import accuracy
from surprise.model_selection import KFold

# 加载数据
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('data/ratings.csv', reader)

# ItemCF 计算得分
# 取最相思的用户计算时，只取最相思的k个
algo = KNNWithZScore(k=40, sim_options={'user_based': False, 'verbose': 'True'})

kf = KFold(n_splits=3)

for train_set, test_set in kf.split(data):
    algo.fit(train_set)
    pred = algo.test(test_set)
    rmse = accuracy.rmse(pred, verbose=True)
    accuracy.mae(pred, verbose=True)
    print(rmse)
