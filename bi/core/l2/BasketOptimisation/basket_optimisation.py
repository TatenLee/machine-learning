# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 这个实在是太难了，不知道数据是如何转换的...
"""

import pandas as pd
from efficient_apriori import apriori

data = pd.read_csv('Market_Basket_Optimisation.csv', header=None, sep='\t', names=['items'])
# 将所有item变为小写
data['items'] = data['items'].str.lower()
data['items'] = data['items'].apply(lambda _: tuple(str(_).split(',')))

data['row_index'] = data.index.values

transactions = list(data['items'].values)

min_support = 0.02
min_confidence = 0.5
item_sets, rules = apriori(transactions, min_support, min_confidence)
print(item_sets, rules)
flag = True
while (len(item_sets) == 0 or len(rules) == 0) and (min_confidence > 0.05 and min_support > 0.001):
    if flag:
        min_support -= 0.01
        flag = False
    else:
        min_confidence -= 0.005
        flag = True
    if min_confidence > 0.05 and min_support > 0.001:
        item_sets, rules = apriori(transactions, min_support, min_confidence)
        print(min_support, min_confidence)

print(f'min_support: {min_support}, min_confidence: {min_confidence}')
print('频繁项集：', item_sets)
print('关联规则：', rules)
