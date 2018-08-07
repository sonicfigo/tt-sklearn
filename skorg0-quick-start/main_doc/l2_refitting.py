# coding=utf-8
"""
没看懂tutorial这个例子
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#refitting-and-updating-parameters

大致意思是改变kernel参数，重新fit后， predict的值是跟着变化的
"""

import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)  # 100 * 10 矩阵
y = rng.binomial(1, 0.5, 100)  # 二项分布

X_test = rng.rand(5, 10)  # 5 * 10 矩阵 用来检测

clf = SVC()  # 默认kernel='rbf'
clf.set_params(kernel='linear').fit(X, y)

print(clf.predict(X_test))
# array([1, 0, 1, 1, 0])

clf.set_params(kernel='rbf').fit(X, y)
print(clf.predict(X_test))
# array([0, 0, 0, 1, 0])
