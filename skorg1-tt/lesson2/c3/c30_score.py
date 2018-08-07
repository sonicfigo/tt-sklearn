# coding=utf-8
"""
每个estimator都有一个score方法评估分数。
"""

from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data  # 共1797条
y = digits.target

svc = svm.SVC(C=1, kernel='linear')

# 手工隔离出100个，作为test数据，用于最后验证，计算分数。
score = svc.fit(X[:-100], y[:-100]).score(X[-100:], y[-100:])
print(score)


