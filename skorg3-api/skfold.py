# coding=utf-8
"""
StratifiedKFold

可以看到 StratifiedKFold
分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)

num_splitting = skf.get_n_splits(X, y)  # 这两个参数 X,y 没什么用
print('num_splitting: %s' % num_splitting)  # 2

print('skf: %s' % skf)

print('\n===================')
for train_index, test_index in skf.split(X, y):
    print
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

"""
StratifiedKFold

('TRAIN:', array([1, 3]))
('TEST:', array([0, 2]))

('TRAIN:', array([0, 2]))
('TEST:', array([1, 3]))
"""

"""
KFold

('TRAIN:', array([2, 3]))
('TEST:', array([0, 1]))

('TRAIN:', array([0, 1]))
('TEST:', array([2, 3]))
"""
