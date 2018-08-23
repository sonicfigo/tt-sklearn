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
"""

# Create a signal with only 2 useful dimensions
import numpy as np
from sklearn import decomposition

x1 = np.random.normal(size=100)  # Mean 是0，Standard deviation 是 1， 100个点
x2 = np.random.normal(size=100)
x3 = x1 + x2  # 此处意思应该是， 这个x3，应该是要被废弃的feature，因为都是来自x1，x2

# X = np.c_[x1, x2, x3]  # (100, 3)
X = np.c_[x3, x1, x2]  # (100, 3)

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
print(pca.explained_variance_)  # [  3.47326399e+00   8.75238725e-01   2.45161260e-32]
print(pca.explained_variance_ratio_)

# As we can see, only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)  # (100, 2)
print('\n===================origin vs reduce')
print(X[0])
print(X_reduced[0])
