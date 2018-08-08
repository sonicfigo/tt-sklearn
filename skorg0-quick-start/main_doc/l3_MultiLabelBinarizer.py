# coding=utf-8
"""
进行mutillabel分类时，训练数据的类别标签Y应该是一个矩阵，
第[i,j]个元素指明了第j个类别标签是否出现在第i个样本数据中。

例如，np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])，这样的一条数据，指明针对
    - 第一条样本数据，类别标签是第0个类，
    - 第二条数据，类别标签是第1，第2个类，
    - 第三条数据，没有类别标签。

有时训练数据中，类别标签Y可能不是这样的可是，
而是类似[[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]这样的格式，
每条数据指明了每条样本数据对应的类标号。
这就需要将Y转换成矩阵的形式，sklearn.preprocessing.MultiLabelBinarizer提供了这个功能。

"""
from sklearn.preprocessing import MultiLabelBinarizer

y_origin = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]

y = MultiLabelBinarizer().fit_transform(y_origin)

print(y)
assert (5, 5) == y.shape
