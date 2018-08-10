# coding=utf-8
"""
Linear model: from regression to sparsity

Linear regression 基础，

2018-08-09 14:55:32 考试出来的分数不高啊，不需要优化吗?
"""
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target,
                                                    test_size=0.045)
print(len(diabetes.data))  # 共有442
print(len(X_train))  # 422
print(len(X_test))  # 20

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.coef_)

print("""
The mean square error
均方差
""")
mean1 = np.mean((regr.predict(X_test) - y_test) ** 2)
print(mean1)

print("""
考试分数， 1 好， 0 差
1 is perfect prediction,
0 means that there is no linear relationship
between X and y.

""")
score1 = regr.score(X_test, y_test)
print(score1)
