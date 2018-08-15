# coding=utf-8
"""
scikit-learn本身内置了一些标准的数据集，例如：
鸢尾花和数字数据集用于分类，还有波士顿房屋价格数据集用来做回归。
接下来用Python 语言来实现加载这些数据集（iris和digits）

数据集1
"""

from sklearn import datasets

iris = datasets.load_iris()  # scikit-learn载入数据集实例:

"""
scikit-learn载入的数据集是以类似于字典的形式存放的，
该对象中包含了所有有关该数据的数据信息（甚至还有参考文献）。
其中的数据值统一存放在.data的成员中，比如我们要将iris数据显示出来，只需显示iris的data成员：

数据都是以n维（n个特征）矩阵形式存放和展现，iris数据中每个实例有4维特征，分别为：
sepal length、
sepal width、
petal length和
petal width。
"""

# 150条记录
print('\niris.data:\n%s' % iris.data)  # 显示iris数据
assert len(iris.data) == len(iris.target) == 150
assert 4 == len(iris.data[0])  # 每一行数据，有4个feature

# 如果是对于监督学习，比如分类问题，数据中会包含对应的分类结果，其存在.target成员中：
print('\niris.target(数字答案):\n%s' % iris.target)

# 三种鸢尾的名字
print('\ntarget_names 存三种鸢尾的名字:%s' % iris.target_names)
assert len(iris.target_names) == 3

print('\ny(数字答案)的对应的鸢尾名字(文字答案)')
iris_names = iris.target_names[iris.target]
print(iris_names)

print(type(iris_names))  # <type 'numpy.ndarray'>
print(iris_names.shape)  # (150,)


