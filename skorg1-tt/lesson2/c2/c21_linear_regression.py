# coding=utf-8
"""
Linear model: from regression to sparsity

Linear regression 基础
"""
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target,
                                                    test_size=0.045)
# print(len(X_train)) # 422
# print(len(X_test)) # 20

model1 = linear_model.LinearRegression()
model1.fit(X_train, y_train)
print(model1.coef_)

"""
The mean square error
均方差
"""
mean1 = np.mean((model1.predict(X_test) - y_test) ** 2)
print(mean1)

"""
考试分数， 1 好， 0 差
1 is perfect prediction,
0 means that there is no linear relationship
between X and y.

"""
score1 = model1.score(X_test, y_test)
print(score1)
