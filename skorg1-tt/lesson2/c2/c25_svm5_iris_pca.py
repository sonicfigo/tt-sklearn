# coding=utf-8
"""
画出的图，与标准答案的颜色标注，对比感受下
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
y = iris.target


def plot_2d():
    """只取前两个 feature 画 2d的点，再根据答案y配上颜色，看到3类，感受一下有没有3个区域"""
    X = iris.data[:, :2]  # we only take the first two features.
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(1, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())


def plot_3d_pca():
    """
    降维成3d， 等于有3个 "不可描述的 feature1,2,3"， 对应 3d 图形的 x,y,z
    再根据答案y配上颜色，看到3类，感受一下有没有3个区域
    """
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(2, figsize=(8, 6))
    ax3d = Axes3D(fig, elev=-150, azim=110)

    pca_n3 = PCA(n_components=3)
    X_reduced = pca_n3.fit_transform(iris.data)
    # 原来可理解的feature 4个， 变成不可描述的feature 3个
    print("plot_3d_pca: %s" % pca_n3.n_components_)
    # 把 3 个feature 对应的3d点，画到ax3d
    ax3d.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
                 cmap=plt.cm.Set1, edgecolor='k', s=40)

    ax3d.set_title("First three PCA directions")

    ax3d.set_xlabel("1st eigenvector")
    ax3d.w_xaxis.set_ticklabels([])

    ax3d.set_ylabel("2nd eigenvector")
    ax3d.w_yaxis.set_ticklabels([])

    ax3d.set_zlabel("3rd eigenvector")
    ax3d.w_zaxis.set_ticklabels([])


def plot_2d_pca():
    """自己尝试：降维成2d，画在平面图什么样子"""
    plt.figure(3, figsize=(8, 6))
    plt.clf()

    pca_n2 = PCA(n_components=2)
    # shape(150, 4) -> shape(150, 1)
    X_reduced = pca_n2.fit_transform(iris.data)  # 注意是PCA的fit_transform
    print("plot_2d_pca: %s" % pca_n2.n_components_)  # 2
    # 把 3 个feature 对应的3d点，画到ax3d
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k',
                s=40)
    plt.xlabel('reduced feature 1')
    plt.ylabel('reduced feature 2')


plot_2d()
plot_3d_pca()
plot_2d_pca()
plt.show()
