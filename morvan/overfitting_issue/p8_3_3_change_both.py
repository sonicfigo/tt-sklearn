# coding=utf-8
"""
之前有个错误版本，本例子为正确方案实现

用 panda 的dataframe 处理， 才能实现目标：k值 和 cv值的混搭得分效果，用3d图形显示
要点，注意以下 3 个数据的 shape ，值，是否正确对应了
- 输入数据 list，
- 输入数据 ndarray
- 输出数据 dataframe

结论：就例子中的range来说， k=10 or 11， cv = 最好
"""
# from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

fig = plt.figure()
ax3d = Axes3D(fig)

iris = load_iris()
X = iris.data
y = iris.target

k_range_list = range(10, 13)  # 准x, 3个k ,     [10, 11, 12]
cv_range_list = range(15, 20)  # 准y, 5个cv,     [5, 6, 7, 8 ,9]

x_k_range, y_cv_range = np.meshgrid(k_range_list, cv_range_list)  # return 2 ndarray
print('\n===================mesh后为（y个数， x个数），既(5, 3)')
print('x.shape:', x_k_range.shape)
print(x_k_range)
print('y.shape:', y_cv_range.shape)
print(y_cv_range)

# 默认分数0, 索引是 cv， column是 k值， 因为要跟 x , y 的shape 一致为(5, 3)
df_score = pd.DataFrame(data=0, index=cv_range_list, columns=k_range_list)  # 要一致为（5, 3）
df_score_empty = df_score.copy()


def _get_accuracy4classification(k, cv):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
    return scores.mean()


for row_idx_is_cv, row_series in df_score.iterrows():
    print('\ncv值-%s' % row_idx_is_cv)  # k
    for col_idx, _ in enumerate(row_series):
        k_value = k_range_list[col_idx]
        print('col-%s: k值-%s' % (col_idx, k_value))

        score = _get_accuracy4classification(k_value, row_idx_is_cv)
        df_score.loc[row_idx_is_cv, k_value] = score  # 基于 label 修改值

print('\n===================当前数据')
print(df_score)

print('\n===================原始数据(5, 3)，既(y个数, x个数)')
print(df_score_empty)


def _draw_3d_and_2d():
    ax3d.plot_surface(x_k_range, y_cv_range, df_score, rstride=1, cstride=1,
                      cmap=plt.get_cmap('rainbow'))

    ax3d.contourf(x_k_range, y_cv_range, df_score, zdir='z', offset=0.962,
                  cmap=plt.get_cmap('rainbow'))

    # 画轮廓线, 无效果
    # C = ax3d.contour(x_k_range, y_cv_range, df_score, 8, colors='black', linewidth=.5)
    # ax3d.clabel(C, inline=True, fontsize=10)


def _beautify_axis_tick():
    plt.xlabel('k_range')
    x_s, x_e = k_range_list[0], k_range_list[-1]
    plt.xticks(np.arange(x_s, x_e, step=1))  # x轴点之间的step 为 1

    plt.ylabel('cv_range')
    y_s, y_e = cv_range_list[0], cv_range_list[-1]
    plt.yticks(np.arange(y_s, y_e, 1))  # y轴点之间的step 为 1


_draw_3d_and_2d()
_beautify_axis_tick()
plt.show()
