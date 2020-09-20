# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 矩阵
"""

from copy import deepcopy
from itertools import chain, product


class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_number):
        """
        得到矩阵的某一行

        :param row_number: 行号 {int} 类型
        :return: 返回矩阵
        """
        return Matrix([self.data[row_number]])

    def col(self, col_number):
        """
        得到矩阵的某一列

        :param col_number:
        :return:
        """
        m = self.shape[0]
        return Matrix([[self.data[i][col_number]] for i in range(m)])

    @property
    def is_square(self):
        """
        判断矩阵是否为方阵

        :return:
        """
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """
        找到原始矩阵的转置

        :return:
        """
        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    @staticmethod
    def _eye(n):
        """
        获取 (n, n) 的单位矩阵

        :param n:
        :return:
        """

        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        """
        获取相同形状的矩阵

        :return:
        """
        assert self.is_square, "The matrix has to be square!"
        data = self._eye(self.shape[0])
        return Matrix(data)

    @staticmethod
    def _gaussian_elimination(aug_matrix):
        """
        简化增广阵(augmented matrix)的左方阵形成对角矩阵

        :param aug_matrix: 增广阵 {list}
        :return:
        """

        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # 从顶部到底部
        for col_index in range(n):
            # 检查对角线上的元素是否为0
            if aug_matrix[col_index][col_index] == 0:
                row_index = col_index
                # 找到对角线上元素不为0，并且元素和列索引一样的行
                while row_index < n and aug_matrix[row_index][col_index] == 0:
                    row_index += 1
                # 将此行添加到对角线上元素的行
                for i in range(col_index, m):
                    aug_matrix[col_index][i] += aug_matrix[row_index][i]

            # 消除非零元素
            for i in range(col_index + 1, n):
                # 跳过零元素
                if aug_matrix[i][col_index] == 0:
                    continue
                # 消除非零元素
                k = aug_matrix[i][col_index] / aug_matrix[col_index][col_index]
                for j in range(col_index, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_index][j]

        # 从底部到顶部
        for col_index in range(n - 1, -1, -1):
            # 消除非零元素
            for i in range(col_index):
                # 跳过零元素
                if aug_matrix[i][col_index] == 0:
                    continue
                # 消除非零元素
                k = aug_matrix[i][col_index] / aug_matrix[col_index][col_index]
                for j in chain(range(i, col_index + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_index][j]

        # 迭代对角线元素
        for i in range(n):
            k = 1 / aug_matrix[i][i]
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    def _inverse(self, data):
        """
        找到矩阵的逆

        :param data:
        :return:
        """
        n = len(data)
        unit_matrix = self._eye(n)
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        res = self._gaussian_elimination(aug_matrix)
        return list(map(lambda _: _[n:], res))

    @property
    def inverse(self):
        """
        获取矩阵的逆

        :return:
        """
        assert self.is_square, "The matrix has to be square!"
        data = self._inverse(self.data)
        return Matrix(data)

    @staticmethod
    def _row_mul(row_a, row_b):
        """
        将两个数组中下标相同的元素相乘并求和

        :param row_a:
        :param row_b:
        :return:
        """
        return sum(x[0] * x[1] for x in zip(row_a, row_b))

    def _mat_mul(self, row_A, B):
        """
        矩阵乘积

        :param row_A:
        :param B:
        :return:
        """
        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        """
        矩阵乘积

        :param B:
        :return:
        """
        error_msg = "A's column count does not match B's row count!"
        assert self.shape[1] == B.shape[0], error_msg
        return Matrix([self._mat_mul(row_a, B) for row_a in self.data])

    @staticmethod
    def _mean(data):
        """
        计算所有样本的平均值

        :param data:
        :return:
        """
        m = len(data)
        n = len(data[0])
        res = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                res[j] += row[j] / m
        return res

    def mean(self):
        """
        计算所有样本的平均值

        :return:
        """
        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """
        放大乘积

        :param scala:
        :return:
        """
        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *= scala
        return Matrix(data)
