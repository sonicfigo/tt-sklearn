# coding=utf-8
"""
input的类型转换
"""

import numpy as np
from sklearn import random_projection, datasets
from sklearn.svm import SVC

"""
指定为float32"""
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
assert np.float32 == X.dtype

"""
input默认都会转为 float64"""
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
assert np.float64 == X_new.dtype
assert float == X_new.dtype

"""
regression 问题是默认转为float64
classification 问题是自定义"""

iris = datasets.load_iris()
clf = SVC()

"""
fit target"""
clf.fit(iris.data, iris.target)
print(list(clf.predict(iris.data[:3])))

"""
fit target名字"""
clf.fit(iris.data, iris.target_names[iris.target])
print(list(clf.predict(iris.data[:3])))
