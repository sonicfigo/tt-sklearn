# coding=utf-8
"""
cross validation 交叉验证1

--------------8.3 n_neighbors = k 就是最好的参数？ 不一定，那么测试一下。

除了换 k， 甚至model也可以换一下。 本例子没有做换model，只是换了scoring

前提： cv = 10
"""
from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

# this is how to use cross_val_score to choose model and configs #
k_range = range(1, 31)  # k到底哪个参数比较好？


def accuracy4classification():
    """
    'accuracy' for classification
    k为13, 18, 20 时候分数最好。k  > 20时， 分数持续降低，是一个overfitting 现象

    """
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        scores_mean = scores.mean()
        k_scores.append(scores_mean)
        # print('\n===================scores_mean:%s' % scores_mean)
        plt.text(k, scores_mean, k, ha='center', va='bottom')  # 把 k 值标注上

    plt.plot(k_range, k_scores)
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
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)

        # 原名为 mean_squared_error，负号， 负值变正值
        loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
        k_scores.append(loss.mean())
    # return k_scores

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated neg_mean_squared_error')
    plt.show()


accuracy4classification()  # accuracy越高越好
# loss4regression()  # loss越低越好
