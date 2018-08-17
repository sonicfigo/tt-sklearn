# coding=utf-8
"""
更好的评估，就需要split training 和 test

名为 KFold 的 cross-validation 理念，

sk learn也包装了此功能 cross_val_score：
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # cv5，进行五次分组

这里为手动实现，模仿这个函数，连 k_fold 函数都不用
"""

import numpy as np
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

model_svc = svm.SVC(C=1, kernel='linear')

print(len(X))
X_folds = np.array_split(X, 3)  # 分成3批
y_folds = np.array_split(y, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on

    X_train = list(X_folds)
    X_test = X_train.pop(k)  # pop 第一批, 599个作为test，
    X_train = np.concatenate(X_train)  # 剩下的两批合并起来做train

    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)

    scores.append(model_svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
