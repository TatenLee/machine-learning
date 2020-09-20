# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description:
"""
import jieba
from jieba import analyse
from jieba import posseg
from textrank4zh import TextRank4Keyword, TextRank4Sentence

with open('data/news_高考.txt', 'r', encoding='utf-8') as file:
    sentence = file.read()
    print(sentence)

    # 获取分词
    seg_list = jieba.cut(sentence, cut_all=False)
    print(' '.join(seg_list))

    # 获取分词和词性
    words = posseg.cut(sentence)
    for word, tag in words:
        print(f'{word}, {tag}')

    # 通过 TF-IDF 获取关键词
    keywords = analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
    print('通过 TF-IDF 进行关键词抽取: ')
    for item in keywords:
        print(f'{item[0]} {item[1]}')

    print('-' * 100)

    # 基于 TextRank 算法的关键词抽取
    keywords = analyse.textrank(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
    print('通过 textrank 进行关键词抽取: ')
    for item in keywords:
        print(f'{item[0]} {item[1]}')

    # 使用 TextRank 输出摘要
    tr4s = TextRank4Sentence()
    tr4s.analyze(sentence, lower=True, source='all_filters')
    print(f'摘要: ')
    for item in tr4s.get_key_sentences(num=1):
        print(f'{item.index} {item.weight} {item.sentence}')

    file.close()
