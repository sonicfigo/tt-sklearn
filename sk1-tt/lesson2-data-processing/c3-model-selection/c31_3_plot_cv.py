# coding=utf-8
"""
svc 遍历使用10个值(10的负十次方 ~ 1)，作为C参数，看score如何变化。

参数C：
C-SVC的惩罚参数C?
默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
"""
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
# 默认底数是10
C_s = np.logspace(start=-10, stop=0, num=10)  # base=-10， 既10的负10次方 ~ 10的0次方，均分10个
print('\n===================使用10个不同C')
print(C_s)

c_score_means = list()
c_score_stds = list()

for C in C_s:
    svc.C = C
    # cv参数为空，None, 既to use the default 3-fold cross validation,
    scores_3fold = cross_val_score(svc, X, y, n_jobs=1)
    # print('%s:  %s' % (C, this_scores))
    c_score_means.append(np.mean(scores_3fold))
    c_score_stds.append(np.std(scores_3fold))

print('\n===================不同C的 cv=3fold 平均值')
print(c_score_means)
print('\n最高平均值')
print(np.max(c_score_means))

print('\n===================不同C的 cv=3fold 标准差')
print(c_score_stds)
print('\n最高标准差')
print(np.max(c_score_stds))


def plot_2d():
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    # 线条1，实线，平均值
    plt.semilogx(C_s, c_score_means)
    # 线条2， 上虚线，平均值 + 标准差
    plt.semilogx(C_s, np.array(c_score_means) + np.array(c_score_stds), 'b--')
    plt.semilogx(C_s, np.array(c_score_means) - np.array(c_score_stds), 'b--')

    locs, labels = plt.yticks()

    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # y轴单位小标

    plt.ylabel('CV score')
    plt.xlabel('Parameter C')
    plt.ylim(0, 1.1)  # 坐标轴范围
    plt.show()


plot_2d()
