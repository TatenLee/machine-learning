# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: hulu
@description: 
"""

import os

import keras
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from ml.core.l9.model_rnn import TextRNN


def dataset_extract(path, filename, num):
    """
    提取部分数据集

    :param path:
    :param filename:
    :param num:
    :return:
    """
    if os.path.exists(path + filename):
        return
    num_cat = {}
    contents, labels = [], []
    with open(path + filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            label, content = line.strip().split('\t')
            if content:
                if label not in num_cat:
                    num_cat[label] = 1
                    contents.append(content)
                    labels.append(label)
                else:
                    if num_cat[label] < num:
                        num_cat[label] = num_cat[label] + 1
                        contents.append(content)
                        labels.append(label)
        f.close()

    with open(path + 'cnews.train.small.txt', 'w', encoding='utf-8', errors='ignore') as f:
        for content, label in zip(contents, labels):
            f.write(f'{label}\t{content}\n')
        f.close()
    print(len(contents))
    print(contents[0])
    print(contents[1])
    print(num_cat)


def read_vocab(vocab_dir):
    """
    读取词汇表

    :param vocab_dir:
    :return:
    """
    with open(vocab_dir, 'r', encoding='utf-8', errors='ignore') as f:
        words = [_.strip() for _ in f.readlines()]
    word_to_id_dict = dict(zip(words, range(len(words))))
    return words, word_to_id_dict


def read_category():
    """
    读取分类目录，固定

    :return:
    """
    category_list = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # category_list = [_ for _ in category_list]
    print(category_list)
    category_to_id_dict = dict(zip(category_list, range(len(category_list))))
    return category_list, category_to_id_dict


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """
    将文件转换为id表示

    :param filename:
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :return:
    """
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])  # 将每句话id化
        label_id.append(cat_to_id[labels[i]])  # 每句话对应的类别的id

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def train_rnn(train_loader):
    model = TextRNN()  # .cuda()
    multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch  # .cuda()
            y = y_batch  # .cuda()
            out = model(x)
            loss = multi_label_soft_margin_loss(out, y)
            print(f'loss = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
            print(accuracy)


if __name__ == '__main__':
    # 1. 提取小数据集
    dataset_extract('data/', 'cnews.train.txt', 100)

    # 2. 数据处理
    # 获取文本的类别及其对应id的字典
    categories, category_to_id = read_category()
    print(categories)

    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab('data/cnews.vocab.txt')

    # 获取字数
    vocab_size = len(words)

    # 3. 数据加载及分批
    # 获取训练数据每个字的id和对应标签的one-hot形式
    x_train, y_train = process_file('data/cnews.train.small.txt', word_to_id, category_to_id, 600)
    print('x_train=', x_train)
    x_val, y_val = process_file('data/cnews.val.txt', word_to_id, category_to_id, 600)

    # cuda = torch.device('cuda')  # 设置 GPU
    x_train, y_train = torch.LongTensor(x_train), torch.Tensor(y_train)
    x_val, y_val = torch.LongTensor(x_val), torch.Tensor(y_val)

    train_dataset = data.TensorDataset(x_train, y_train)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)

    # 4. 模型训练
    train_rnn(train_loader)
