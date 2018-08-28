# coding=utf-8
"""
Principal component analysis: PCA
将能够解释数据信息最大方差的的连续成分提取出来

X，多维数据
C: component，成分
L: loadings，载荷

X = LC，提取成分 C

pca实例关键属性：
.fit()
.explained_variance_

.n_components
. fit_transform()

总结点：
pca.components_ 在 fit后，会变成

shape (n_components 类, 原feature数)
"""

# Create a signal with only 2 useful dimensions
import numpy as np
from sklearn import decomposition

x1 = np.random.normal(size=100)  # Mean 是0，Standard deviation 是 1， 100个点
x2 = np.random.normal(size=100)
x3 = x1 + x2  # 此处意思应该是， 这个x3，应该是要被废弃的feature，因为都是来自x1，x2

X = np.c_[x1, x2, x3]  # (100, 3)，竖向合并，列1都是x1，列2 x2， 列3 x3
print(X)  # (100, 3)

pca = decomposition.PCA()
pca.fit(X)  # 用X填充PCA

"""
第一个是explained_variance_
    它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
第二个是explained_variance_ratio_
    它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。
    所有比例合为1
"""
# TODO 2018-08-22 17:38:00 都是按降序来的？那怎么对比到具体哪个feature。 阅读 PCA & ICA 概念
print('\n===================explained_variance_')
print(pca.explained_variance_)  # [  3.47326399e+00   8.75238725e-01   2.45161260e-32]
print('\n===================explained_variance_ratio_')
print(pca.explained_variance_ratio_)

# As we can see, only the 2 first components are useful
pca.n_components = 2
print("""===================
pca.components_.shape 会从 (3, 3) -> (2, 2)
X 从 (100, 3) -> (100, 2)
""")
print(X.shape)
print(pca.components_.shape)
X_reduced = pca.fit_transform(X)  # (100, 2)
print(pca.components_.shape)
print(X_reduced.shape)

print('\n===================origin vs reduce ，无法观察出，原数据与transform数据的强关联关系')
print(X[0])
print(X_reduced[0])
