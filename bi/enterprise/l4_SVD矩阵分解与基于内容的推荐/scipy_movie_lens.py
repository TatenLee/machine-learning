# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.linalg import svd

if __name__ == '__main__':
    df = pd.read_csv('./data/movielens_sample.txt')
    data = df[['movie_id', 'user_id', 'rating']]

    kf = KFold(n_splits=3)
    for train_set, test_set in kf.split(data):
        A = np.array(data.iloc[train_set])
        p, s, q = svd(A, full_matrices=False)
        print(p)
        print(s)
        print(q)

