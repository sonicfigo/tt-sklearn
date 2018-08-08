# coding=utf-8
"""
cross validation 交叉验证1

交叉验证，对于我们选择正确的 model 和model 的参数是非常有帮助的.
有了他的帮助, 我们能直观的看出不同 model 或者参数对结构准确度的影响.

---------------------------8.1 传统的单一split法(未cross的)
场景：
    单一的 train_test_split 的数据分法，可能造成偏差，因为学习和考试的数据是固定独立分开的。

方案：
    对 training data 和 test data 的多种分法，得到每组分法的考试分数
    综合每组分数，得到最终分数，更为完整，不会偏差太多。
"""

from __future__ import print_function

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

# test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
print(X_train.shape)
print(X_test.shape)

knn = KNeighborsClassifier(n_neighbors=5)  # 找数据点附近的5个点的值,综合后得到y prediction
knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))
