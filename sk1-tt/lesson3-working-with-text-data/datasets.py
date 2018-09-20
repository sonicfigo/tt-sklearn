# coding=utf-8
"""
取数据工具类
"""
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


def load_train_4kinds_newsgp():
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories, shuffle=True,
                                      random_state=42)
    return twenty_train


def load_test_4kinds_newsgp():
    twenty_test = fetch_20newsgroups(subset='test',
                                     categories=categories, shuffle=True,
                                     random_state=42)
    return twenty_test
