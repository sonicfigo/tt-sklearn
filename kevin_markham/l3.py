# coding=utf-8
"""
l3. famouse iris dataset

classification：有限的无序值
regression：连续的有序值

X   =   features = data     =   矩阵matrix
y   =   response = target   =   向量vector


"""

from sklearn.datasets import load_iris

iris = load_iris()
print("""
sklearn 的 features 和 target 类型要求：""")

print("""
1. features 和 response 一定是分开存储的。

2&3. 不管是分类还是回归问题，features 和 response 都必须是 numeric 组成的 ND array。""")
print(type(iris.data))  # <type 'numpy.ndarray'>
print(type(iris.target))  # <type 'numpy.ndarray'>

print("""
4. 特定的矩阵
    - feature一定是二维的，1st维是row * 2nd维是feature, 如150（rows） * 4（features）。
    - target 一定是一维的，且一定等于 feature的1st维的量级（150rows）。""")
print(iris.data.shape)  # (150, 4)
print(iris.target.shape)  # (150,)

print('其他对象类型')
print(type(iris))  # <class 'sklearn.datasets.base.Bunch'>
print(type(iris.feature_names))  # <type 'list'>
print(type(iris.target_names))  # <type 'numpy.ndarray'>
