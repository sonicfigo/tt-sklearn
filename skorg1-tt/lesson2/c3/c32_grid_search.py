# coding=utf-8
"""
Grid-search and cross-validated estimators

sklearn 定义了一种object，如 GridSearchCV

它可以对 estimator 的某个参数，给定一个 parameter grid，找到 score 最大的parameter。

这种 object 的构造函数，接收:
    - 1个 estimator，并具有 estimator 的同样接口，如fit，score ...
    - 1个 param_grid


"""
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
# Cs = np.logspace(start=-10, stop=0, num=10)
Cs = np.logspace(-6, -1, 10)

# By default, the GridSearchCV uses a 3-fold cross-validation.
svc_gridsearch = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)

print("""
以下3个是 train 部分的数据
""")
svc_gridsearch.fit(X[:1000], y[:1000])
print(svc_gridsearch.best_score_)  # 0.925...
print(svc_gridsearch.best_estimator_.C)  # 0.0077...

print("""
以下是当做正常 estimator 来用，执行 score

******************* 引出一个问题，此时这个模型，使用的 C 是多少，查文档可得：

GridSearchCV implements a “fit” and a “score” method.

The parameters of the estimator used to apply these methods 
are optimized by cross-validated grid-search over a parameter grid.

既是：不是普通的 fit 和 score，而是每次操作，会被优化，所以无法正确的知道，到底一次score用了什么参数
所以，想plot一个3d图看效果：cs的变动，对应score分数的变动，是做不到的

之前有 validation_curve 例子，可以做到：
/Users/figo/pcharm/ml/tt-sklearn/morvan/overfitting_issue/l10.py
""")
# Prediction performance on test set is not as good as on train set
print(svc_gridsearch.score(X[1000:], y[1000:]))  # 0.943...
print(svc_gridsearch)

print('\n===================所有结果')
print(svc_gridsearch.cv_results_)
