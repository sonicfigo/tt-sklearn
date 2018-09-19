# coding=utf-8
"""
kfold 和 skfold 有什么区别到底？

挑出的 test index
kfold 是顺序的
[0 1]
[2 3]
[4 5]
[6 7]

skfold 是跳跃的
[0 2]
[1 3]
[4 6]
[5 7]

按文档说的是：
可以看到StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

X = np.array([
    [1, 2, 3, 4],
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34],
    [41, 42, 43, 44],
    [51, 52, 53, 54],
    [61, 62, 63, 64],
    [71, 72, 73, 74]
])

y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
# n_folds这个参数没有，引入的包不同，

print('\n=================== KFold')
kfold = KFold(n_splits=4, random_state=0, shuffle=False)
for train, test in kfold.split(X, y):
    print('Train: %s | test: %s' % (train, test))
    print(" ")

print('\n=================== StratifiedKFold')

skfold = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)
for train, test in skfold.split(X, y):
    print('Train: %s | test: %s' % (train, test))
    print(" ")
