# coding=utf-8
"""
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#refitting-and-updating-parameters

大致意思是改变kernel参数，重新fit后， predict的值是跟着变化的

不需要细看predict 为何出来那些结果，只需知道参数是可以变动的即可
"""

import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)

X = rng.rand(100, 10)  # 100 * 10 矩阵
y = rng.binomial(1, 0.5, 100)  # 二项分布
X_test = rng.rand(5, 10)  # 5 * 10 矩阵 用来检测

clf = SVC()  # 默认kernel='rbf'

print('\n===================X_test')
print(X_test)
clf.set_params(kernel='linear').fit(X, y)  # SVC()构造后，修改kernel，首次fit
print(clf.predict(X_test))
# array([1, 0, 1, 1, 0])

clf.set_params(kernel='rbf').fit(X, y)  # 之前fit和predict过，可以继续修改kernel，再次fit
print('\n===================X_test')
print(X_test)
print(clf.predict(X_test))
# array([0, 0, 0, 1, 0])

