# coding=utf-8
"""
分别用
    - knn
    - linear model
解决digit 分类问题，knn分数稍高
"""

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.1)

knn = neighbors.KNeighborsClassifier()
linear_logistic = linear_model.LogisticRegression()

print("""
KNN %s""" % knn.fit(X_train, y_train).score(X_test, y_test))
print("""
线性逻辑回归 %s""" % linear_logistic.fit(X_train, y_train).score(X_test, y_test))
