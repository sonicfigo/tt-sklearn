# coding=utf-8
"""
查找哪个 alpha 更好
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = \
    train_test_split(diabetes.data, diabetes.target, test_size=0.045)

print(len(X_train))  # 422个
print(len(X_test))  # 20个

regr = linear_model.Ridge()  # Ridge

alphas = np.logspace(-4, -1, 6)  # 10的-4和-1次方，6个等比数据
print("""
遍历这6个alpha去尝试
%s""" % alphas)

plt.figure()
score_list = []
for alpha in alphas:
    regr.set_params(alpha=alpha)
    regr.fit(X_train, y_train)
    print('\n===================coef')
    print(regr.coef_)
    score = regr.score(X_test, y_test)
    score_list.append(score)

    plt.text(alpha + 0.005, score + 0.001, '%s:%.8f' % ('alpha', alpha), ha='center',
             va='bottom')

print("""
6个alpha的得分
%s，\n最低分: %s, 最高分: %s""" % (score_list, np.min(score_list), np.max(score_list)))

print('\n===================看图形，随着alpha的上升，分数时而下降，时而上升的，这怎么评价 alpha 的取值标准？')

plt.plot(alphas, score_list)  # 画线
plt.scatter(alphas, score_list, s=3)

plt.show()
