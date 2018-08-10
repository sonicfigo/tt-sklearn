# coding=utf-8
"""
Ridge + alpha，缩减了特征，将 coef 趋向0，表现更好了，六条拟合线形状相似

且体现了几点：
    1. alpha 是 与 bias 成正比。越高，bias越高（拟合线，都没有穿过点）
    2. bias 越高，variance 越低 （拟合线之间，都是长得比较像的）
"""
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.c_[.5, 1].T
y_list = [.5, 1]
X_test_fixed = np.c_[0, 2].T

"""
the larger the ridge alpha parameter,
the higher the bias and the lower the variance.

文档说到:当 alpha 越高 （留待下一图形再验证）
    - bias 越高，既训练数据的表现越差 (注意看，线并没有完全穿过两点, 且6个线长得很相似)
    - variance 越低，既真实数据的表现越好 (因为整个 model 很稳定)

"""
# egr = linear_model.LinearRegression() # 之前是用这个，没有任何 shrink 作用，6次拟合的结果，差别大
regr = linear_model.Ridge(alpha=.1)  # alpha 是关键

# 画出平面图看看，注意观测 12个点 和 6条线 的关系
plt.figure()

np.random.seed(0)
for _ in range(6):
    X_random_each = .1 * np.random.normal(size=(2, 1)) + X
    regr.fit(X_random_each, y_list)
    print('\n===================coef')
    print(regr.coef_)
    plt.scatter(X_random_each, y_list, s=3)

    y_predict_each = regr.predict(X_test_fixed)
    plt.scatter(X_test_fixed, y_predict_each, s=31)
    plt.plot(X_test_fixed, y_predict_each)

plt.show()
