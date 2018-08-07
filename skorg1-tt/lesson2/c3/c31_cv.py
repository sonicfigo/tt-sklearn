# coding=utf-8
"""
模拟 cross_val_score 的过程
"""

from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')

"""
The cross-validation can then be performed easily:
不需要再手工KFlod了
"""
k_fold = KFold(n_splits=3)
scores = [
    svc.fit(X_digits[train], y_digits[train])
        .score(X_digits[test], y_digits[test])
    for train, test in k_fold.split(X_digits)
    ]
print(scores)

print("""
以上步骤可以浓缩成一个函数 cross_val_score:
n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.

""")

cv_scores = cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
print(cv_scores)
