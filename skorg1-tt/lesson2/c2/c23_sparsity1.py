# coding=utf-8
"""
遍历留个alphas，找出最优的那个，重新fit，可以观察到，10个coef里面，有一些趋于0的，被稀疏化了。
(不太理解，因为分数相对高的时候，如0.6,0.7时，coef并没有典型的特征，如很多0。)
"""
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
import numpy as np

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target,
                                                    test_size=0.045)

alphas = np.logspace(-4, -1, 6)
regr = linear_model.Lasso()
scores = [
    regr.set_params(alpha=alpha)
        .fit(X_train, y_train)
        .score(X_test, y_test)
    for alpha in alphas
    ]
best_score = scores.index(max(scores))
best_alpha = alphas[best_score]
regr.alpha = best_alpha
regr.fit(X_train, y_train)

print('所有分数：%s' % scores)
print('最优分数：%s' % scores[best_score])
print('double check：%s' % regr.score(X_test, y_test))
print
print('此时的coefficient为:%s' % regr.coef_)
