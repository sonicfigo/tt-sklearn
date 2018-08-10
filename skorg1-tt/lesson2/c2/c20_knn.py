# coding=utf-8
"""
Supervised learning: predicting an output variable from high-dimensional observations

Nearest neighbor and the curse of dimensionality
知识点：
    - KNN fit ， predict
    - train_test_split

"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

y_unique = np.unique(iris_y)  # y总共就是3个值：0, 1, 2

iris_X_train, iris_X_test, iris_y_train, iris_y_test = \
    train_test_split(iris_X, iris_y, test_size=0.1)

"""
使用 KNN model
Create and fit a nearest-neighbor classifier
"""
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

y_predict = knn.predict(iris_X_test)
print(y_predict)  # 预测

print(iris_y_test)  # 真实答案


# TODO 未细读 "维度灾难"
"""
The curse of dimensionality
For an estimator to be effective, you need the distance between neighboring points to be less than some value d, which depends on the problem. 

In one dimension, this requires on average n \sim 1/d points. In the context of the above k-NN example, if the data is described by just one feature with values ranging from 0 to 1 and with n training observations, then new data will be no further away than 1/n. Therefore, the nearest neighbor decision rule will be efficient as soon as 1/n is small compared to the scale of between-class feature variations.

If the number of features is p, you now require n \sim 1/d^p points. Let’s say that we require 10 points in one dimension: now 10^p points are required in p dimensions to pave the [0, 1] space. As p becomes large, the number of training points required for a good estimator grows exponentially.

For example, if each point is just a single number (8 bytes), then an effective k-NN estimator in a paltry p \sim 20 dimensions would require more training data than the current estimated size of the entire internet (±1000 Exabytes or so).

This is called the curse of dimensionality and is a core problem that machine learning addresses.
"""
