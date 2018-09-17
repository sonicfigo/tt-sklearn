# coding=utf-8
"""
Plot different SVM classifiers in the iris dataset

4种 model 的，针对 iris 的精简数据( 4个feature -> 2个feature )
画出的分界线 decision boundaries (注意！！！！！！不是 有margin 的那个 decision function)

区别：
#### LinearSVC
- 最小化       squared hinge loss
- 多类别分类    one-vs-all

#### SVC(kernel='linear')
- 最小化       regular hinge loss
- 多类别分类    one-vs-one

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

print(__doc__)


# 画底层
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1  # -1 和 +1 是留出空地，好看
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  # 花萼长 x_min ~ x_max，每隔0.02一个
                         np.arange(y_min, y_max, h))  # 花萼宽 y_min ~ y_max，每隔0.02一个
    return xx, yy


# 画等高图
def plot_contours(ax, clf, xx, yy, **params):
    """
    画 decision boundaries，就是底部有三种颜色，既是三类的等高线

    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z_predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # 打平 xx,yy,并把两矩阵左右相加
    Z_predict = Z_predict.reshape(xx.shape)  # (61600,) -> (220, 280)

    # 填充等高色，contour filling color
    out = ax.contourf(xx, yy, Z_predict, **params)  # xx, yy 都是 (220, 280)
    return out


iris = datasets.load_iris()
X = iris.data[:, :2]  # 只取前两个feature 来测试
y = iris.target

# We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub_axs = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 调整图形大小而已

# X0， feature 1，花萼长
# X1， feature 2，花萼宽
X_length, X_width = X[:, 0], X[:, 1]

# 画底图的散点图， xx， yy shape都为 (220, 280)
mesh_xx_length, mesh_yy_width = make_meshgrid(X_length, X_width)

for clf, title, sub_ax in zip(models, titles, sub_axs.flatten()):
    plot_contours(sub_ax, clf, mesh_xx_length, mesh_yy_width,
                  cmap=plt.cm.coolwarm,
                  alpha=0.8,
                  )

    # 画两个 feature 对应的点
    sub_ax.scatter(X_length, X_width, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    sub_ax.set_xlim(mesh_xx_length.min(), mesh_xx_length.max())
    sub_ax.set_ylim(mesh_yy_width.min(), mesh_yy_width.max())
    sub_ax.set_xlabel('Sepal length')
    sub_ax.set_ylabel('Sepal width')
    sub_ax.set_xticks(())
    sub_ax.set_yticks(())
    sub_ax.set_title(title)

plt.show()
