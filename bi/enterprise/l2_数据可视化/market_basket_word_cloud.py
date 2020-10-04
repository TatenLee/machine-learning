# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 词云展示
"""

import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def remove_stop_words(data, stop_words):
    """
    去除停用词

    :param data:
    :param stop_words:
    :return:
    """
    for stop_word in stop_words:
        data = data.replace(stop_word, '')
    return data


def create_word_cloud(data, word_cloud_name, target_path='./data'):
    """
    生成词云

    :param data:
    :param word_cloud_name:
    :param target_path:
    :return:
    """
    print('根据词项，开始生成词云...')
    data = remove_stop_words(data, list())
    # 使用 nltk 进行分词
    cut_text = ' '.join(word_tokenize(data))
    wc = WordCloud(
        max_words=100,
        width=2000,
        height=1200
    )
    word_cloud = wc.generate(cut_text)
    # 写词云图片
    word_cloud.to_file(f'{target_path}/{word_cloud_name}')
    # 显示词云文件
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()


def main():
    # 数据加载
    data = pd.read_csv('./data/Market_Basket_Optimisation.csv', header=None)

    # 将数据存放到 transactions 中
    transactions = list()
    item_count = dict()
    for i in range(0, data.shape[0]):
        transaction = list()
        for j in range(0, data.shape[1]):
            item = str(data.values[i, j])
            if item != 'nan':
                transaction.append(item)
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
        transactions.append(transaction)

    # 生成词云
    all_word = ' '.join(f'{transaction}' for transaction in transactions)
    create_word_cloud(all_word, 'market_basket_word_cloud.jpg')

    # 输出 TOP 10 商品
    print(sorted(item_count.items(), key=lambda _: _[1], reverse=True))


if __name__ == '__main__':
    main()
