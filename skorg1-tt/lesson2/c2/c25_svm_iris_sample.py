# coding=utf-8
"""
SVM Exercise

A tutorial exercise for using different SVM kernels.
This exercise is used in the :ref:`using_kernels_tut` part of the
:ref:`supervised_learning_tut` section of the :ref:`stat_learn_tut_index`.

plt画图部分复杂繁琐，没看通
"""
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)
    #
    # Circle out the test data,圆圈里的是test数据
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    #
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    print(y_min, '', y_max)
    #
    # XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    # Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    #
    # # Put the result into a color plot
    # Z = Z.reshape(XX.shape)
    # plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    # plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
    #             levels=[-.5, 0, .5])
    #
    # plt.title(kernel)
plt.show()
