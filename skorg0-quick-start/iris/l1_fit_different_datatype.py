# coding=utf-8
"""
学习与输出的类型对应关系
"""

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()

clf = SVC()
clf.fit(iris.data, iris.target)  # 学习的是数字
print(list(clf.predict(iris.data[:3])))  # 输出数字 [0, 0, 0]

clf.fit(iris.data, iris.target_names[iris.target])  # 学习的是名字
print(list(clf.predict(iris.data[:3])))  # 输出名字 ['setosa', 'setosa', 'setosa']
