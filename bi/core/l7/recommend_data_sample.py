# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 将数据采样为小样本
"""

# 设置 max_size
max_size = 100000

size = 0
with open('data/sample.csv', 'wb') as writer:
    with open('data/tianchi_fresh_comp_train_user.csv', 'rb') as file:
        while size <= max_size:
            line = file.readline()
            writer.write(line)
            size += 1
