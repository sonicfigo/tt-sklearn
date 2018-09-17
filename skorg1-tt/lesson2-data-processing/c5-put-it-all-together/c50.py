# coding=utf-8
"""
Pipelining 基本概念

estimator 侧重点不同，有的 transform data，有的 predict variables，可以组合使用

Pipelining，就是用来将多个学习器组成流水线，通常为：
将数据标准化的学习器---特征提取的学习器---执行预测的学习器

本例子：
- 建一个pipeline， 由两个model，pca 和 lr 组成
- 挑选两个 model 的参数选择范围，用于 GridSearchCV, gscv
- gscv fit了 train 数据后，会生成最优的 best_estimator_
- 本例结束，并没有进而把 gscv 拿来做 predict。待详见后续c51例子

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

"""
pipeline 打包步骤：
1. pca降维
2. logistic 线性回归
"""

digits = datasets.load_digits()  # (1797, 64)
X_digits = digits.data
y_digits = digits.target


def fill_pipeline():
    plt.figure(1)
    plt.clf()
    # plt.axes([.2, .2, .7, .7])

    m1_pca = decomposition.PCA()
    """
    fit 与否
    只影响 explained_variance_ 生成和展示
    对最终 gscv 的 best n_components 选择没影响 
    (猜测：因为gscv自己最终也要fit)
    
    PS:
    但是当真gscv的要predict时，要先 fit pca过的数据，
    此时需要pca做transform，那会就会报错，所以pca最终还是要fit的
    此例子没有做 predict
    """
    m1_pca.fit(X_digits)

    # Plot the PCA spectrum
    print('\n===================原始 estimator')
    print('pca 的explained_variance_，表达64个 feature 各自的重要性')
    print('有64个variance，因为img有64个feature')
    print('m1_pca.explained_variance_.shape:')
    print(m1_pca.explained_variance_.shape)  # shape (64, 0)

    """
    plot 参数1若是 len=n 的 list，那么这个list做y轴
    且x轴默认为[0,1,2,3,4,5,6...n]，刚好与 n_components 概念一致，所以做x轴刚好
    """
    plt.plot(m1_pca.explained_variance_, linewidth=2)

    # 上一行代码plot(y)，相当于下方代码plot(x,y)
    # plt.plot(range(64), m1_pca.explained_variance_, linewidth=2)

    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    m2_logistic = linear_model.LogisticRegression()
    # Pipeline 自身也实现了 BaseEstimator 接口
    return Pipeline(steps=[('pca', m1_pca),
                           ('logistic', m2_logistic)])


def fit_gscv(pipe):
    """
    Parameters of pipelines can be set using ‘__’ separated parameter names:
    - pca的n_components参数
    - lr的C参数
    """
    # Prediction
    n_components = [20, 40, 64]

    # 10的-4次方 ~ 10的4次方，既是 0.0001 ~ 1000，等比选3个，既[0.0001, 1, 10000]
    Cs = np.logspace(-4, 4, 3)
    gscv_pipe = GridSearchCV(pipe, dict(pca__n_components=n_components,
                                        logistic__C=Cs))

    print('\n===================gscv 要 fit完，才有对应的best estimator')
    gscv_pipe.fit(X_digits, y_digits)
    return gscv_pipe


def show_best(gscv):
    print("\n======== pipeline's 自动选择的 best estimator")

    print('\n"pca 的explained_variance_（前n个feature，因为best自动选了n_components=n）"')
    best_pca = gscv.best_estimator_.named_steps['pca']
    print('explained_variance_.shape')
    print(best_pca.explained_variance_.shape)  # (40, 0)

    best_logistic = gscv.best_estimator_.named_steps['logistic']

    # 画一个说明文字
    plt.axvline(best_pca.n_components,
                linestyle=':',
                label='best: n_components-%s, C-%s' %
                      (best_pca.n_components, best_logistic.C))
    plt.legend(prop=dict(size=12))


pipe = fill_pipeline()
gscv = fit_gscv(pipe)
show_best(gscv)

"""
pipe.named_steps:

'pca': 
PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
svd_solver='auto', tol=0.0, whiten=False) 

'logistic': 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
"""
print('\n===================pipe named steps，原始model，所以参数都是原始的，如 n_componnets=None)')
print(pipe.named_steps)

print('\n===================gscv的，参数自动 best 了 如 n_componnets = 40 ')
print(gscv.best_estimator_.named_steps)

"""
pipeline 要使用，也是要fit的
之前 gscv 的fit是用来找寻最优参数的
"""
print('\n===================score')
pipe.fit(X_digits, y_digits)
print(pipe.score(X_digits, y_digits))

plt.show()
