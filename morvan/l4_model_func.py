# coding=utf-8
"""
KNeighborsClassifier

load                数据
train_test_split    分开数据
fit                 学习
predict             预测
score               看分数
"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # 选择临近的几个点，综合后模拟出预测数据值

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print('\n===================取前两行，所有列')
print(iris_X[:2, :])
print(iris_y[:2])

print('\n===================分出学习和考试数据(test占30%)，并且打乱顺序')
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 学习

print('\n预测 vs 真实')
print('预测')
y_predict = knn.predict(X_test)
print(y_predict)  # 预测
print('真实：')
print(y_test)  # 真实值

print('\n===================正确率')
print(knn.score(X_test, y_test))

print('\n===================正确与否的矩阵，验证了与正确率相符')
print(y_predict == y_test)
print(type(y_predict))  # <type 'numpy.ndarray'>
print(len(y_predict))
