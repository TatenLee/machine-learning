# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import torch.nn.functional as func
from torch import nn


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(5000, 64)  # 进行词嵌入
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.8),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 10),
                                nn.Softmax())

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = func.dropout(x, p=0.8)
        x = self.f1(x[:, -1, :])
        return self.f2(x)
