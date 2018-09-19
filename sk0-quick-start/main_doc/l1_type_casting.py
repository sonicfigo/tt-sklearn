# coding=utf-8
"""
输入  float32
转成
输出  float64
"""

import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)  # 10 *2000的矩阵
X = np.array(X, dtype='float32')
print('输入类型 %s' % X.dtype)  # dtype('float32')

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print('自动转换后，输出类型 %s' % X_new.dtype)  # dtype('float64')
