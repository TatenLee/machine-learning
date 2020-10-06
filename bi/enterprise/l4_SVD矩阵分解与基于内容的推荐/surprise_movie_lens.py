# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

from surprise import accuracy, Dataset, Reader, SVD
from surprise.model_selection import KFold

if __name__ == '__main__':
    # 读取数据
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    data = Dataset.load_from_file('./data/ratings.csv', reader=reader)
    train_set = data.build_full_trainset()

    # 使用 funkSVD
    algo = SVD(biased=False)

    # 定义 K 折交叉验证迭代器
    kf = KFold(n_splits=3)
    for train_set, test_set in kf.split(data):
        # 训练并预测
        algo.fit(train_set)
        predictions = algo.test(test_set)

        # 计算 RMSE
        accuracy.rmse(predictions, verbose=True)

    uid = str(196)
    iid = str(302)

    # 输出 uid 对 iid 的预测结果
    algo.predict(uid, iid, r_ui=4, verbose=True)
