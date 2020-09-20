# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 用 pytorch 预测波士顿房价
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# 1. 加载数据
data = load_boston()
X = data['data']
y = data['target']
print(y.shape)
y = y.reshape(-1, 1)
print(y)

# 数据规范化
ss = MinMaxScaler()
X = ss.fit_transform(X)

# 数据集切分
X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.FloatTensor)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)

# 2. 构建网络
model = nn.Sequential(
    # 输入的是13，10是神经元的个数（隐层为10）
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 3. 定义优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练
max_epoch = 1000
iter_loss = []

for i in range(max_epoch):
    # 对输入的 X 进行预测
    y_pred = model(X)
    # 计算 loss
    loss = criterion(y_pred, y)
    # 因为 loss 只有一个值
    iter_loss.append(loss.item())
    # 清空上一轮梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 调整权重
    optimizer.step()

# 5. 测试
output = model(test_x)
predict_list = output.detach().numpy()
print(predict_list)

# 6. 绘制图像
# 6.1 绘制不同的 iteration 的 loss
x = np.arange(max_epoch)
y = np.array(iter_loss)
plt.plot(x, y)
plt.title('Loss value in all iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean loss value')
plt.show()

# 6.2 绘制真实值与与预测值的散点图
x = np.arange(test_x.shape[0])
y_pred = np.array(predict_list)
y_test = np.array(test_y)
# 预测用红色
line_pred = plt.scatter(x, y_pred, c='red')
# 真实值用蓝色
line_real = plt.scatter(x, y_test, c='blue')
plt.legend([line_pred, line_real], ['predict', 'real'], loc=1)
plt.title('Prediction VS Real')
plt.ylabel('House Price')
plt.show()
