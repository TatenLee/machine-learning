# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 对 delicious2k 数据进行推荐
原始数据集：https://grouplens.org/datasets/hetrec-2011/
数据格式：userID     bookmarkID     tagID     timestamp
"""

import logging
import math
import operator
import random

import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [line:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

global user_items, user_tags, item_users, item_tags, tag_users, tag_items


def load_data(path):
    """
    加载数据

    :param path:
    :return:
    """
    records = dict()
    logging.info('开始加载数据...')
    df = pd.read_csv(path, sep='\t')
    for i in range(len(df)):
        uid = df['userID'][i]
        iid = df['bookmarkID'][i]
        tid = df['tagID'][i]

        # 键不存在时，设置默认值
        records.setdefault(uid, {})
        records[uid].setdefault(iid, [])
        records[uid][iid].append(tid)
    logging.info(f'数据集大小为: {len(df)}\n设置tag的人数: {len(records)}')
    return records


def train_test_split(data, ratio, version=100):
    """
    划分训练集和测试集

    :param data:
    :param ratio: 测试集比例
    :param version:
    :return:
    """
    train_data = dict()
    test_data = dict()

    random.seed(version)
    for u in data.keys():
        for i in data[u].keys():
            if random.random() < ratio:
                test_data.setdefault(u, {})
                test_data[u].setdefault(i, [])
                for t in data[u][i]:
                    test_data[u][i].append(t)
            else:
                train_data.setdefault(u, {})
                train_data[u].setdefault(i, [])
                for t in data[u][i]:
                    train_data[u][i].append(t)
    logging.info(f'训练集样本数: {len(train_data)}, 测试集样本数: {len(test_data)}')
    return train_data, test_data


def create_connection_matrix(mat, index, item, value=1):
    """
    创建关系矩阵

    :param mat:
    :param index:
    :param item:
    :param value:
    :return:
    """
    if index not in mat:
        mat.setdefault(index, {})
        mat[index].setdefault(item, value)
    else:
        if item not in mat[index]:
            mat[index][item] = value
        else:
            mat[index][item] += value


def create_connection(data):
    """
    创建关系

    :param data:
    :return:
    """
    global user_items, user_tags, item_users, item_tags, tag_users, tag_items
    user_items = dict()
    user_tags = dict()
    item_users = dict()
    item_tags = dict()
    tag_users = dict()
    tag_items = dict()

    for u, items in data.items():
        for i, tags in items.items():
            for t in tags:
                create_connection_matrix(user_items, u, i, 1)
                create_connection_matrix(user_tags, u, t, 1)
                create_connection_matrix(item_users, i, u, 1)
                create_connection_matrix(item_tags, i, t, 1)
                create_connection_matrix(tag_users, t, u, 1)
                create_connection_matrix(tag_items, t, i, 1)

    logging.info(f'user_tags 大小: {len(user_tags)}, tag_items 大小: {len(tag_items)}, user_items 大小: {len(user_items)}')

    return user_tags, tag_items, user_items


def recommend(user, N, method='simple'):
    """
    对用户推荐 Top-N

    :param user:
    :param N:
    :param method: simple, norm, idf
    :return:
    """
    recommend_items = dict()

    # 对 item 进行打分，分数为所有的（用户对某标签使用的次数 wut * 商品被打上相同标签的次数 wti）之和
    tagged_items = user_items[user]

    for tag, wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            if item in tagged_items:
                continue

            norm = 1
            if method == 'simple':
                norm = 1
            elif method == 'norm':
                norm = len(tag_users[tag].items()) * len(user_tags[user].items())
            elif method == 'idf':
                norm = math.log(len(tag_users[tag].items()) + 1)

            if item not in recommend_items:
                recommend_items[item] = wut * wti / norm
            else:
                recommend_items[item] = recommend_items[item] + wut * wti / norm

    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0:N]


def precision_and_recall(train_data, test_data, N, method):
    """
    使用测试集计算准确率和召回率

    :param train_data:
    :param test_data:
    :param N:
    :param method:
    :return:
    """
    hit = 0
    h_recall = 0
    h_precision = 0
    for user, items in test_data.items():
        if user not in train_data:
            continue
        # 获取 Top-N 推荐列表
        rank = recommend(user, N, method)
        for item, rui in rank:
            if item in items:
                hit = hit + 1
        h_recall = h_recall + len(items)
        h_precision = h_precision + N
    return hit / (h_precision * 1.0), hit / (h_recall * 1.0)


def main():
    # 保存 user 对 item 的 tag，即 {user_id: {item1: [tag1, tag2], ...}}
    records = load_data('./data/user_taggedbookmarks-timestamps.dat')
    logging.info('加载数据完成...')

    # 划分训练集和测试集
    train_data, test_data = train_test_split(records, 0.2)

    # 使用训练集构建 user_tags, tag_items, user_items
    global user_tags, tag_items, user_items
    user_tags, tag_items, user_items = create_connection(train_data)

    # 使用 tagBasedIDF 进行推荐
    method = 'idf'

    # 使用测试集对推荐结果进行评估
    for n in [5, 10, 20, 40, 60, 80, 100]:
        precision, recall = precision_and_recall(train_data, test_data, n, method)
        logging.info(f'{n} precision: {precision * 100:.4f}%, recall: {recall * 100:.4f}%')


if __name__ == '__main__':
    main()
