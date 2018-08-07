# coding=utf-8
"""
更新参数后的refitting
"""
import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)  # shape(100, 10)

# print(y)

X_test = rng.rand(5, 10)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
print(clf.predict(X_test))

clf.set_params(kernel='rbf').fit(X, y)  # 更改参数后，predict值变了
print(clf.predict(X_test))
