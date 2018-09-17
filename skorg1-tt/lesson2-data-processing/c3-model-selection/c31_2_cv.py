# coding=utf-8
"""
Cross-validation generators

cv generators 暴露一个 split 方法， KFold 就是这种 generators 其中之一

介绍了 cross_val_score ，并先用 kfold 繁琐的实现了相同效果，纯粹用于解释原理
"""

from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

model_svc = svm.SVC(C=1, kernel='linear')

"""
演示了 cv generators 的 split用法，搭配上 fit 和 scores
"""
k_fold = KFold(n_splits=3)
scores = [
    model_svc
        .fit(X[train], y[train])
        .score(X[test], y[test])
    for train, test in k_fold.split(X)
]
print(scores)

print("""
以上步骤可以浓缩成一个函数 cross_val_score:
n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.

""")

cv_scores = cross_val_score(model_svc, X, y, cv=k_fold, n_jobs=-1)
print(cv_scores)
