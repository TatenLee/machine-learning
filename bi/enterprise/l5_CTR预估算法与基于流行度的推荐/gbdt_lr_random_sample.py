# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: GBDT+LR算法Demo演示
随机生成二分类样本8万个，每个样本20个特征
采用RF，RF+LR，GBDT，GBDT+LR进行二分类预测
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def rf(X_train, X_test, y_train, y_test):
    """
    RF

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=3)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    return fpr_rf, tpr_rf


def gbdt(X_train, X_test, y_train, y_test):
    """
    GBDT

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    gbdt = GradientBoostingClassifier(n_estimators=n_estimator)
    gbdt.fit(X_train, y_train)
    y_pred_gbdt = gbdt.predict_proba(X_test)[:, 1]
    fpr_gbdt, tpr_gbdt, _ = roc_curve(y_test, y_pred_gbdt)
    return fpr_gbdt, tpr_gbdt


def lr(X_train, X_test, y_train, y_test):
    """
    LR

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    lr = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    return fpr_lr, tpr_lr


def rf_lr(X_train, X_test, y_train, y_test):
    """
    RF + LR

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    # 基于随机森林的监督变换
    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=3)
    rf.fit(X_train, y_train)
    # 得到 OneHot 编码
    rf_enc = OneHotEncoder(categories='auto')
    rf_enc.fit(rf.apply(X_train))
    # 使用 OneHot 编码作为特征，训练 LR
    rf_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    rf_lr.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    # 使用 LR 进行预测
    y_pred_rf_lr = rf_lr.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr_rf_lr, tpr_rf_lr, _ = roc_curve(y_test, y_pred_rf_lr)
    return fpr_rf_lr, tpr_rf_lr


def gbdt_lr(X_train, X_test, y_train, y_test):
    """
    GBDT + LR

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    # 基于 GBDT 的监督变换
    gbdt = GradientBoostingClassifier(n_estimators=n_estimator)
    gbdt.fit(X_train, y_train)
    # 得到 OneHot 编码
    gbdt_enc = OneHotEncoder(categories='auto')
    np.set_printoptions(threshold=np.inf)
    gbdt_enc.fit(gbdt.apply(X_train)[:, :, 0])
    # 使用 OneHot 编码作为特征，训练 LR
    gbdt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    gbdt_lr.fit(gbdt_enc.transform(gbdt.apply(X_train_lr)[:, :, 0]), y_train_lr)
    y_pred_gbdt_lr = gbdt_lr.predict_proba(gbdt_enc.transform(gbdt.apply(X_test)[:, :, 0]))[:, 1]
    fpr_gbdt_lr, tpr_gbdt_lr, _ = roc_curve(y_test, y_pred_gbdt_lr)
    return fpr_gbdt_lr, tpr_gbdt_lr


if __name__ == '__main__':
    n_estimator = 10

    # 生成样本集
    X, y = make_classification(n_samples=80000, n_features=20)

    # 将样本集分成测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # 再将训练集拆成两个部分（GBDT/RF, LR）
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

    # 使用 RF
    fpr_rf, tpr_rf = rf(X_train, X_train_lr, y_train, y_train_lr)

    # 使用 GBDT
    fpr_gbdt, tpr_gbdt = gbdt(X_train, X_train_lr, y_train, y_train_lr)

    # 使用 LR
    fpr_lr, tpr_lr = lr(X_train, X_train_lr, y_train, y_train_lr)

    # 使用 RF + LR
    fpr_rf_lm, tpr_rf_lr = rf_lr(X_train, X_train_lr, y_train, y_train_lr)

    # 使用 GBDT + LR
    fpr_gbdt_lr, tpr_gbdt_lr = gbdt_lr(X_train, X_train_lr, y_train, y_train_lr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_gbdt, tpr_gbdt, label='GBDT')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_rf_lm, tpr_rf_lr, label='RF + LR')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBDT + LR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_gbdt, tpr_gbdt, label='GBDT')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_rf_lm, tpr_rf_lr, label='RF + LR')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBDT + LR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (zoomed)')
    plt.legend(loc='best')
    plt.show()
