# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 使用surprise进行预测
其中ALS和SGD是两种不同的优化方式
BaselineOnly是预测的方式
    对于Baseline来讲，最终求解一个r_ui值，这个值等于所有用户对商品评分的均值+用户的偏好值+商品的平均表现值
    将r_ui和最终预测的结果进行比对
"""

from surprise import Dataset, Reader, BaselineOnly, KNNBasic, NormalPredictor, accuracy
from surprise.model_selection import KFold

# 1. 读取数据
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('data/ratings.csv', reader=reader)

# 2. 选择模型
# ALS优化
# bsl_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}
# SGD优化
bsl_options = {'method': 'sgd', 'n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)

# 3. 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for train_set, test_set in kf.split(data):
    # 4. 训练并预测
    algo.fit(train_set)
    predictions = algo.test(test_set)
    # 5. 计算RMSE
    accuracy.rmse(predictions, verbose=True)

uid = str(196)
iid = str(302)

# 输出uid对iid的预测结果
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
