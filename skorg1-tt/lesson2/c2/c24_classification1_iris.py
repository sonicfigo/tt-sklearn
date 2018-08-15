# coding=utf-8
"""
linear regression - 线性回归 不适用分类问题，要用
logistic regression - 逻辑回归
"""

import numpy as np
from sklearn import datasets, linear_model

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X))  # 随机打乱

iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

"""
Shrinkage and sparsity with logistic regression

The C parameter controls the amount of regularization in the LogisticRegression object: 
a large value for C results in less regularization. 

penalty="l2" gives Shrinkage (i.e. non-sparse coefficients), 
while penalty="l1" gives Sparsity.


C 越大，regularization 越小

penalty 
- 'L2'，shrink（non-sparse）(default)，压缩，接近零，不会减小feature数
- 'L1'，sparsity，稀疏，可以到0，减小feature数
"""
model_lr = linear_model.LogisticRegression(C=1e5)

""""
model详情，注意看multi_class 是 one-versus-all
LogisticRegression(C=100000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
"""
print(model_lr)

model_lr.fit(iris_X_train, iris_y_train)
print(model_lr.score(iris_X_test, iris_y_test))
