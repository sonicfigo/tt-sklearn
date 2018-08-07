# coding=utf-8
"""
model  常用属性和功能
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
X = loaded_data.data
y = loaded_data.target
model = LinearRegression()
model.fit(X, y)

print('\n===================以此为例子：y = 0.1x + 0.3')

print('\n=================== 系数，既Θ，既0.1')
print('\n 本例子多个θ，既意思这些影响房价的系数值各为多少：面积、楼层、地段等等')
print(model.coef_)

print('\n=================== 与y轴的交点， 既0.3')
print(model.intercept_)

print('\n=================== model 默认参数')
print(model.get_params())

print('\n===================model 学习成果的打分')
print("""
score 打分机制：
regression 是 R^2 coefficient of determination

如果是classification的model：
打分机制就是 真实数据 和 预测数据 的正确度的百分比
""")
print(model.score(X, y))
