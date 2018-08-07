# coding=utf-8
"""

"""

from sklearn import linear_model

reg = linear_model.LinearRegression()
"""
feature1 和 feature2 线性函数的结果：
00得0
11得1
22得2
"""
reg.fit(
    [[0, 0],
     [1, 1],
     [2, 2]],

    [0, 1, 2])
"""
2个feature，所以coef也是两个
"""
print(reg.coef_)
