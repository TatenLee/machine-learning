# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""
import os
import pickle
from _collections import defaultdict

import jieba
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer


def split_text(text):
    """
    利用 jieba 进行分词

    :param text:
    :return:
    """
    text = text.replace(' ', '').replace('\n', '')
    text_words = jieba.cut(text.strip())
    res = ' '.join([word for word in text_words if word not in stopwords])
    return res


def find_similar_text(cp_index, top_n=10):
    """
    查找相似文本

    :param cp_index: 嫌疑文章索引
    :param top_n: 与前多少进行对比。默认前10
    :return:
    """
    s = class_id[id_class[cp_index]]
    print(s)
    # 只在新华社发布的文章中查找
    dist_dict = {_: cosine_similarity(tfidf[cp_index], tfidf[_]) for _ in class_id[id_class[cp_index]]}
    # 从大到小进行排序
    return sorted(dist_dict.items(), key=lambda _: _[1][0], reverse=True)[:top_n]


# 1. 记录终止词，避免影响文章语义
with open('data/chinese_stopwords.txt', 'r', encoding='UTF-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]

# 2. 加载内容数据集
news = pd.read_csv('data/sqlResult.csv', encoding='GB18030')
print(news.shape)  # 89611, 7
print(news.head(5))

# 2.1 处理缺失值
news = news.dropna(subset=['content'])
print(news.shape)  # 87054, 7
print(news.iloc[0].content)  # 查看第一行内容
print(split_text(news.iloc[0].content))

# 2.2 加载进行过分词的数据集
if not os.path.exists('data/corpus.pkl'):
    # 如果数据不存在则进行加载数据集，并且
    # 对数据集中的 content 进行分词
    corpus = list(map(split_text, [str(_) for _ in news.content]))
    print(corpus[0])
    with open('data/corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file)
else:
    # 如果数据存在则直接进行使用
    with open('data/corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)

print('加载数据集完成...')

# 3. 得到corpus的TF-IDF矩阵
count_vectorizer = CountVectorizer(encoding='GB18030', min_df=0.015)  # min_df 代表大于百分之1.5词频的单词才会参会与计算
tfidf_transformer = TfidfTransformer()

# 3.1 给定模型
count_vector = count_vectorizer.fit_transform(corpus)
tfidf = tfidf_transformer.fit_transform(count_vector)

# 3.2 标记是否为自己的新闻
label = list(map(lambda _: 1 if '新华' in str(_) else 0, news.source))

# 3.3 切分数据集
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3)

# 4. 利用朴素贝叶斯去进行分类预测是否为自己的新闻
# 4.1 模型训练
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 4.2 模型预测
# y_predict = clf.predict(X_test)
prediction = clf.predict(tfidf.toarray())  # 预测全量
labels = np.array(label)
compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': labels})
print('分类预测完成...')

# 5. 取出预测是新华社，但实际不是的列索引
# 和实际为新华社的新闻
copy_news_index = compare_news_index[
    (compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index

xinhuashe_news_index = compare_news_index[(compare_news_index['labels'] == 1)].index

# 6. 对 tfidf 数据进行标准化
normalizer = Normalizer()
scaled_array = normalizer.fit_transform(tfidf.toarray())
print('标准化完成...')

# 7. 使用 k-means 对全量文档进行聚类
if not os.path.exists('data/label.pkl'):
    kmeans = KMeans(n_clusters=25)
    k_labels = kmeans.fit_predict(scaled_array)
    print('聚类完成...')

    # 7.1 保存聚类结果
    with open('data/label.pkl', 'wb') as file:
        pickle.dump(k_labels, file)
    print('已保存聚类结果...')

    # 7.2 保存 id-class
    id_class = {index: class_ for index, class_ in enumerate(k_labels)}

    with open('data/id_class.pkl', 'wb') as file:
        pickle.dump(id_class, file)
else:
    with open('data/label.pkl', 'rb') as file:
        k_labels = pickle.load(file)
    with open('data/id_class.pkl', 'rb') as file:
        id_class = pickle.load(file)

class_id = defaultdict(set)
for index, class_ in id_class.items():
    # 只统计新华社发布的class_id
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)
    with open('class_id.pkl', 'wb') as file:
        pickle.dump(class_id, file)

# 在 copy_news_index 里面找一个
cp_index = 3352
similar_list = find_similar_text(cp_index)
print(similar_list)
print(f'怀疑抄袭\n{news.iloc[cp_index].content}')

# 找一篇相似的原文
similar = similar_list[0][0]
print(f'相似原文\n{news.iloc[similar].content}')
