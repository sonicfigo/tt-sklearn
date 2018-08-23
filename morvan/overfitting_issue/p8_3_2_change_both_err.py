# coding=utf-8
"""
让 k_range 和 cv_range 都变化，根据 3d数据，看哪个组合分数最高

注意：本例子是一个错误示范，盲目的认为 reshape 后的数据，就会正确对应 k_range 和 cv_range的组合

"""
from __future__ import print_function

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

fig = plt.figure()
ax3d = Axes3D(fig)

iris = load_iris()
X = iris.data
y = iris.target

k_range_list = range(10, 13)
cv_range_list = range(15, 17)

k_range, cv_range = np.meshgrid(k_range_list, cv_range_list)  # return 2 ndarray

print('\n===================origin k & cv')
print(k_range_list)
print(cv_range_list)

print('\n===================after mesh k')
print(k_range)
print('\n===================after mesh cv')
print(cv_range)


def accuracy4classification():
    kcv_score_list = []  # k 与 cv 搭配，得到的分数
    for k in k_range_list:
        for cv in cv_range_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
            scores_mean = scores.mean()
            kcv_score_list.append(scores_mean)
            print('k-%s, cv-%s, scores_mean-%s' % (k, cv, scores_mean))

    kcv_score_nda = np.array(kcv_score_list)  # (25,0)
    print('kcv_score_nda')
    print(kcv_score_nda)

    raise Exception('注意这句, 错误的，问题就是在这句 shape后的score并不能正确对应原 x，y的组合')
    nd_kcv_scores55 = kcv_score_nda.reshape(k_range.shape)

    print('\n===================scores')
    print(nd_kcv_scores55)

    ax3d.plot_surface(k_range, cv_range, nd_kcv_scores55,
                      rstride=1,
                      cstride=1,
                      cmap=plt.get_cmap('rainbow')
                      )

    # 填充等高色，无效果
    # ax3d.contourf(k_range, cv_range, nd_kcv_scores55, zdir='z', offset=-0.5,
    #               cmap=plt.get_cmap('rainbow'))

    plt.xlabel('k_range')
    x_s, x_e = k_range_list[0], k_range_list[-1]
    plt.xticks(np.arange(x_s, x_e, step=1))  # y轴点之间的step 为 1

    plt.ylabel('cv_range')
    y_s, y_e = cv_range_list[0], cv_range_list[-1]
    plt.yticks(np.arange(y_s, y_e, 1))  # y轴点之间的step 为 1

    plt.show()


accuracy4classification()  # accuracy越高越好
