# coding=utf-8
"""
学习：数字 对应 中文
预测：中文 by 数字
"""
from sklearn import svm

clf = svm.SVC()

features_list = [[1], [2], [3], [1], [1], [1], [1], [1]]  # list of features
targets = ['一', '二', '三', '一', '壹', '壹', '壹', '一']
clf.fit(X=features_list, y=targets)

for predict in clf.predict([[1], [3]]):123
    print predict
