# coding=utf-8
"""

"""

from __future__ import print_function
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target,
                                                    test_size=0.045)
print(len(X_train))

regr = linear_model.Ridge()

alphas = np.logspace(-4, -1, 6)
print("""
遍历这6个alpha去尝试
%s""" % alphas)

score_list = [
    regr.set_params(alpha=alpha)
        .fit(X_train, y_train, )
        .score(X_test, y_test)
    for alpha in alphas
    ]
print("""
6个alpha的得分
%s""" % score_list)
