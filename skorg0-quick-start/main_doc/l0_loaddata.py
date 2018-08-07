# coding=utf-8
"""
基础load姿势
"""

from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print(len(iris.data))
print(len(digits.data))
