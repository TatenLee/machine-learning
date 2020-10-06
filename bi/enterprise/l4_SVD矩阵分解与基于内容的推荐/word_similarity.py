# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""
import multiprocessing
import os

import jieba
from gensim.models import word2vec


def segment_depart(sentence):
    """
    利用 jieba 进行分词

    :param sentence:
    :return:
    """
    words = jieba.cut(sentence.strip())
    return ' '.join(words)


def get_or_create_cut_word(input_path, output_path):
    """
    获取分词文件

    :param input_path:
    :param output_path:
    :return:
    """
    if os.path.exists(output_path):
        return word2vec.PathLineSentences(output_path)
    else:
        input_data = open(input_path, 'r')
        output_data = open(output_path, 'w')
        for line in input_data:
            output_data.write(segment_depart(line) + '\n')
        return word2vec.PathLineSentences(output_path)


def get_or_train_model(model_path):
    """
    训练模型

    :param model_path:
    :return:
    """
    if os.path.exists(model_path):
        return word2vec.Word2Vec.load(model_path)
    else:
        model = word2vec.Word2Vec(sentences, size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())
        model.save(model_path)
        return model


if __name__ == '__main__':
    # 读取数据
    sentences = get_or_create_cut_word('./data/three_kingdoms.txt', './data/three_kingdoms_cut.txt')

    # 设置模型参数，进行训练
    model = get_or_train_model('./data/three_kingdoms.model')

    vector = model.wv.get_vector('曹操') - model.wv.get_vector('刘备') + model.wv.get_vector('刘玄德')
    print(model.wv.similar_by_vector(vector))
