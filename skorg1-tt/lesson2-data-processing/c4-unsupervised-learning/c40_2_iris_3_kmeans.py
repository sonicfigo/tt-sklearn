# coding=utf-8
"""
=========================================================
K-means Clustering
=========================================================

3 k-means 不同参数下的聚类图形：
-   8类聚类
The next plot displays what using eight clusters would deliver

-   3类聚类
The plots display firstly what a K-means algorithm would yield
using three clusters.

-   bad initiliazation， n_init = 1
It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.

和最终真实应该有的情况

and finally the ground truth.

"""

import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cluster
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
"""
[0]'sepal length (cm)'
[1]'sepal width (cm)'
[2]'petal length (cm)'
[3]'petal width (cm)'
"""
X = iris.data
y = iris.target

fignum = 1


def plot_3_kinds():
    global fignum
    kmeans_list = [('k_means_iris_8', cluster.KMeans(n_clusters=8)),
                   ('k_means_iris_3', cluster.KMeans(n_clusters=3)),
                   # 没觉得这个有多bad，聚类的结果与真实结果，及n_init未更改的model，还比较相似啊
                  ('k_means_iris_bad_init', cluster.KMeans(n_clusters=3,
                                                           n_init=1,
                                                           init='random'))]

    titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
    for name, model_kmeans in kmeans_list:
        fig = plt.figure(fignum)
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        model_kmeans.fit(X)

        print('\n===================labels_ 既是等于 predict(X)? 结论：是的')
        labels = model_kmeans.labels_

        """
        x轴 petal width 
        y轴 sepal length
        z轴 petal length
        """
        print(labels)
        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        # 修饰图形
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12

        fignum = fignum + 1


def plot_ground_truth():
    """
    真实答案应该有的图形
    """
    fig = plt.figure(fignum)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    for name, target_idx in [('Setosa', 0),
                             ('Versicolour', 1),
                             ('Virginica', 2)]:
        # 画文字， x,y,z 位置，放在这个类X的均值处，显示起来好看
        ax.text3D(X[y == target_idx, 3].mean(),
                  X[y == target_idx, 0].mean(),
                  X[y == target_idx, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

    """
    Reorder the labels to have colors matching the cluster results
    根据答案画颜色
    
    choose 把原来的 【0，0，..., 1, 1, ..., 2, 2, ...】 替换成 [1, 1, ..., 2, 2, ...., 0, 0,..]
    只是为了和原始的y答案的颜色排列，错开，看起来更明显，是ground truth的
    """
    y_choose = np.choose(y, [1, 2, 0]).astype(np.float)
    # 画点, c=color
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y_choose, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')

    ax.set_title('Ground Truth')
    ax.dist = 12


plot_3_kinds()
plot_ground_truth()
plt.show()
