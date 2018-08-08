# coding=utf-8
"""
模仿p8_3的思路，研究，固定k值的情况下，变动 cv 分组数，到底哪个cv值最好

前提： k = 5

结论是大概 17， 18 组时，分数最高
"""
from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

cv_range = range(5, 31)


def accuracy4classification():
    """
    'accuracy' for classification
    k为12 ~ 18时候考试分数最好。k  > 18时， 分数降低，是一个overfitting 现象
    """
    k_scores = []
    for cv in cv_range:
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        k_scores.append(scores.mean())

    plt.plot(cv_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def loss4regression():
    """
    'mean_squared_error'(新版本改为neg_mean_squared_error)
    for regression， 判断预测值和真实值的误差是多少， 在回归里，会比accuracy好

    结果同样是 13 ~18 效果最好(误差最小)
    """
    k_scores = []
    for cv in cv_range:
        knn = KNeighborsClassifier(n_neighbors=5)

        # 原名为 mean_squared_error，负号， 负值变正值
        loss = -cross_val_score(knn, X, y, cv=cv, scoring='neg_mean_squared_error')
        k_scores.append(loss.mean())
    # return k_scores

    plt.plot(cv_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated neg_mean_squared_error')
    plt.show()


accuracy4classification()  # accuracy越高越好
loss4regression()  # loss越低越好
