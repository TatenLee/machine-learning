# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用ALS进行矩阵分解
"""

from ml.core.l4.ALS import ALS


def load_movie_ratings(file_name):
    """
    加载数据集

    :param file_name:
    :return:
    """
    f = open(file_name)
    lines = iter(f)
    col_names = ', '.join(next(lines)[:-1].split(',')[:-1])
    print(f'The column names are: {col_names}')
    data = [[float(r) if i == 2 else int(r) for i, r in enumerate(line[:-1].split(',')[:-1])] for line in lines]
    f.close()
    return data


def format_prediction(item_id, score):
    """
    格式化预测结果

    :param item_id:
    :param score:
    :return:
    """
    return f'item_id: {item_id}, score: {score:.2f}'


# 1. 加载数据
X = load_movie_ratings('data/ratings_small.csv')

# 2. 选择模型
model = ALS()

# 3. 模型训练
model.fit(X, k=3, max_iter=2)

print('对用户进行推荐')
user_ids = range(1, 13)
predictions = model.predict(user_ids, n_items=2)
print(predictions)
for user_id, prediction in zip(user_ids, predictions):
    _prediction = [format_prediction(item_id, score) for item_id, score in prediction]
    print(f'user_id: {user_id}, recommendation: {_prediction}')
