# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

from collections import defaultdict

import numpy as np

from ml.core.l4.Matrix import Matrix


class ALS(object):
    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None

    def _process_data(self, X):
        """
        将评分矩阵X转化为稀疏矩阵

        :param X: {list} -- 2d list with int or float(user_id, item_id, rating)
        :return:
            dict -- {user_id: {item_id: rating}}
            dict -- {item_id: {user_id: rating}}
        """
        self.user_ids = tuple((set(map(lambda _: _[0], X))))
        self.user_ids_dict = dict(map(lambda _: _[::-1], enumerate(self.user_ids)))

        self.item_ids = tuple((set(map(lambda _: _[1], X))))
        self.item_ids_dict = dict(map(lambda _: _[::-1], enumerate(self.item_ids)))

        self.shape = (len(self.user_ids), len(self.item_ids))

        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating

        err_msg = f'用户的数量为: {len(self.user_ids)}, 评分的数量为: {len(ratings)}, 数量不匹配'
        assert len(self.user_ids) == len(ratings), err_msg

        err_msg = f'物品的数量为: {len(self.item_ids)}, 评分的数量为: {len(ratings)}, 数量不匹配'
        assert len(self.item_ids) == len(ratings_T), err_msg

        return ratings, ratings_T

    def _users_mul_ratings(self, users, ratings_T):
        """
        用户矩阵（稠密）与评分矩阵（稀疏）相乘

        :param users: {Matrix} -- k * m matrix, m stands for number of user_ids.
        :param ratings_T: {dict} -- The items ratings by users.
                        {item_id: {user_id: rating}}
        :return: Matrix -- Item matrix.
        """

        def f(users_row, item_id):
            user_ids = iter(ratings_T[item_id].keys())
            scores = iter(ratings_T[item_id].values())
            col_numbers = map(lambda _: self.user_ids_dict[_], user_ids)
            _users_row = map(lambda _: users_row[_], col_numbers)
            return sum(a * b for a, b in zip(_users_row, scores))

        res = [[f(users_row, item_id) for item_id in self.item_ids] for users_row in users.data]
        return Matrix(res)

    def _items_mul_ratings(self, items, ratings):
        """
        item矩阵（稠密）与评分矩阵（稀疏）相乘

        :param items: {Matrix} -- k * n matrix, n stands for number of item_ids.
        :param ratings: {dict} -- The items ratings by users.
                        {user_id: {item_id: rating}}
        :return: Matrix -- User matrix.
        """

        def f(items_row, item_id):
            item_ids = iter(ratings[item_id].keys())
            scores = iter(ratings[item_id].values())
            col_numbers = map(lambda _: self.item_ids_dict[_], item_ids)
            _items_row = map(lambda _: items_row[_], col_numbers)
            return sum(a * b for a, b in zip(_items_row, scores))

        res = [[f(items_row, user_id) for user_id in self.user_ids] for items_row in items.data]
        return Matrix(res)

    def _gen_random_matrix(self, n_rows, n_cols):
        """
        生成随机矩阵

        :param n_rows:
        :param n_cols:
        :return:
        """
        data = np.random.rand(n_rows, n_cols)
        return Matrix(data)

    def _get_rmse(self, ratings):
        """
        计算RMSE

        :param ratings:
        :return:
        """
        m, n = self.shape
        mse = 0.0
        n_elements = sum(map(len, ratings.values()))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                if rating > 0:
                    user_row = self.user_matrix.col(i).transpose
                    item_col = self.item_matrix.col(j)
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse ** 0.5

    def fit(self, X, k, max_iter=10):
        """
        模型训练

        :param X:
        :param k:
        :param max_iter:
        :return:
        """
        ratings, ratings_T = self._process_data(X)
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        m, n = self.shape

        error_msg = 'k值必须小于原始矩阵的秩'
        assert k < min(m, n), error_msg

        self.user_matrix = self._gen_random_matrix(k, m)
        for i in range(max_iter):
            if i % 2:
                items = self.item_matrix
                self.user_matrix = self._items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings
                )
            else:
                users = self.user_matrix
                self.item_matrix = self._users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings_T
                )
            rmse = self._get_rmse(ratings)
            print(f'Iterations: {i + 1}, RMSE: {rmse:.6f}')

        self.rmse = rmse

    def _predict(self, user_id, n_items):
        """
        Top-n 推荐，用户列表: user_id, n_items: Top-n

        :param user_id:
        :param n_items:
        :return:
        """
        users_col = self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose

        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        items_scores = map(lambda _: (self.item_ids[_[0]], _[1]), items_col)
        viewed_items = self.user_items[user_id]
        items_scores = filter(lambda _: _[0] not in viewed_items, items_scores)

        return sorted(items_scores, key=lambda _: _[1], reverse=True)[:n_items]

    def predict(self, user_ids, n_items=10):
        """
        预测多个用户

        :param user_ids:
        :param n_items:
        :return:
        """
        return [self._predict(user_id, n_items) for user_id in user_ids]
