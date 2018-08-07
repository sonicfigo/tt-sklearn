# coding=utf-8
"""
Scikit-Learn 7 normalization 标准化数据，既是scale
"""

from sklearn import preprocessing
import numpy as np

a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]], dtype=np.float64)

print('有无scale的数据集对比')

print(a)
print('')
print(preprocessing.scale(a))


