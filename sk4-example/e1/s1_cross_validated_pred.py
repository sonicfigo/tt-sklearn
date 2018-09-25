# coding=utf-8
"""
scatter 画出 target 和 cross_val_predict 做出的predict 的对应关系

有没有趋向那条 plot出来的 线性函数？
"""

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X = boston.data
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
"""
画点
"""
ax.scatter(y, predicted, edgecolors=(0, 0, 0))

"""
画线
x轴范围 y.min-5 ~ y.max-50
y轴范围 y.min-5 ~ y.max-50

相当于一条直线，起点(5,5)，终点(50,50)
"""
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
print('\n===================')
plt.show()
