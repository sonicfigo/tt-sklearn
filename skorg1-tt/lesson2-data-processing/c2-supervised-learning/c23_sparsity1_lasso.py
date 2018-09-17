# coding=utf-8
"""
遍历6个alphas，找出最优的那个
重新fit，可以观察到，10个coef里面，有一些趋于0的，被稀疏化了。
(不太理解，因为分数相对高的时候，如0.6,0.7时，coef并没有典型的特征，如很多0。)
"""
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target,
                                                    test_size=0.045)

alphas = np.logspace(-4, -1, 6)
regr = linear_model.LassoCV()  # Lasso(least absolute shrinkage and selection operator)
print (regr)
score_list = []
for alpha in alphas:
    regr.set_params(alpha=alpha)
    regr.fit(X_train, y_train)
    score_each = regr.score(X_test, y_test)

    print('\n===================分数 %s' % score_each)
    print('coef')
    print(regr.coef_)
    score_list.append(score_each)

print('\n所有分数：\n%s' % score_list)

best_score_idx = score_list.index(max(score_list))
print('\n最优分数：%s' % score_list[best_score_idx])

best_alpha = alphas[best_score_idx]
print('\n最优分数对应alpha：%s' % best_alpha)

regr.set_params(alpha=best_alpha)
regr.fit(X_train, y_train)
print('\ndouble check：%s' % regr.score(X_test, y_test))

print('\n此时的coefficient为:%s' % regr.coef_)
