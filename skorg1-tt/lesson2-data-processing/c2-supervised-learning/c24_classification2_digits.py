# coding=utf-8
"""
两种方式，解决digit 分类问题，
- knn
knn分数稍高

- linear model
文档说到：
linear regression is not the right approach as it will give too much weight to data
far from the decision frontier.

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, neighbors, linear_model, svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

digits = datasets.load_digits()  # data (1797, 64)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.1)


def by_KNN():
    print(X_train.shape)  # (1617, 64)
    print('KNN')
    model_knn = neighbors.KNeighborsClassifier()
    print(model_knn)
    print('分数 %s' % model_knn.fit(X_train, y_train).score(X_test, y_test))


def by_LR():
    print('逻辑回归')
    model_linear_logistic = linear_model.LogisticRegression()
    print(model_linear_logistic)
    print('分数 %s' % model_linear_logistic.fit(X_train, y_train).score(X_test, y_test))


def by_SVM():
    print('\n===================提前尝试，下一章提到的SVM')
    print('SVM 竟然分数那么高，除了rbf核以外')

    for fig_num, kernel in enumerate(['linear', 'rbf', 'poly']):
        # for fig_num, kernel in enumerate(['linear']):
        model = svm.SVC(kernel=kernel)
        model.fit(X_train, y_train)
        print('[%-6s] 分数 %s' % (kernel, model.score(X_test, y_test)))

        plot_SMV_PCA(fig_num)


def plot_SMV_PCA(fig_num):
    """
    用iris PCA 的例子模仿一下， 画出digits的pca
    """
    fig = plt.figure(fig_num)
    ax3d = Axes3D(fig)

    # 画 train 点
    X_train_reduced = PCA(n_components=3).fit_transform(X_train)
    ax3d.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], X_train_reduced[:, 2],
                 c=y_train, cmap=plt.cm.Set1, edgecolor='k', s=40)

    # 画 test 点，圆圈里的是test数据
    X_test_reduced = PCA(n_components=3).fit_transform(X_test)
    ax3d.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], X_test_reduced[:, 2],
                 c=y_test, cmap=plt.cm.Set1,
                 edgecolor='k', s=500, facecolors='none', zorder=10)


# by_KNN()
# by_LR()
by_SVM()
plt.show()

"""
plot 一下 digits 的 svm 的 图，看linear 这么准确，分出来的10类到底会是怎么样？
马上遇到一个坎，x轴和y轴，只能取2维数据，那digit是64维的，如何plot出来可视化。
涉及到高维数据的降维等知识点

plot_SMV_PCA 完不成了
"""
