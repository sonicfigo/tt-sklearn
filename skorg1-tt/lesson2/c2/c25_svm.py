# coding=utf-8
"""

"""

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

"""
kernel='linear', Linear kernel，线性分界。
还可以用
='poly'，Polynomial kernel, 多边形分界。
='rbf', Radial Basis Function
"""
svc = svm.SVC(kernel='linear')
iris_X_train, _, iris_y_train, _2 = train_test_split(iris.data, iris.target,
                                                     test_size=0.1)
svc.fit(iris_X_train, iris_y_train)

print(svc)
