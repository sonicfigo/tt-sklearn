# coding=utf-8
"""
KFold split的原理：
数据有6条，split 分成3份，每次1份（2条）作为测试。
"""

from sklearn.model_selection import KFold, cross_val_score


X = ['a', 'a', 'b', 'c', 'c', 'c']
k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))
