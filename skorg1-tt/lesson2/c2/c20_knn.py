# coding=utf-8
"""
Supervised learning: predicting an output variable from high-dimensional observations

Nearest neighbor and the curse of dimensionality
KNN的一个简单例子，其中涉及 train_test_split的原理，不知何意

"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y,
                                                                        test_size=0.1)

"""
使用 KNN model
Create and fit a nearest-neighbor classifier
"""
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

y_ = knn.predict(iris_X_test)
print(y_)  # 预测

print(iris_y_test)  # 真实答案
