# coding=utf-8
"""

"""
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


def load_train():
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories, shuffle=True,
                                      random_state=42)
    return twenty_train
