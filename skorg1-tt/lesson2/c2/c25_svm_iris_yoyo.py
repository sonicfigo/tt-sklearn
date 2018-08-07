# coding=utf-8
"""
根据 头2个feature， 区分 iris数据中的 1类 和 2类

4个feature，预测3种类别，分数更高，学得更好。
2个feature，预测2种类别，分数不高，不稳定。


"""

# import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def predict_2_by_feature_2():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 0, :2]  # 只要 y = 1/2 的对应的X， 且X只要前两个feature
    y = y[y != 0]

    assert len(X) == len(y) == 100  # 共150个，去掉y=0的50个

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model_svc = svm.LinearSVC().fit(X_train, y_train)
    y_ = model_svc.predict(X_test)
    print(y_)
    print(y_test)

    print(model_svc.score(X_test, y_test))


def predict_3_by_feature_4():
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        test_size=0.1)

    model_svc = svm.LinearSVC().fit(X_train, y_train)
    y_ = model_svc.predict(X_test)
    print(y_)
    print(y_test)

    print(model_svc.score(X_test, y_test))

print('少feature，预测2类，分数低。')
predict_2_by_feature_2()

print('多feature，预测3类，分数高。')
predict_3_by_feature_4()
