# coding=utf-8
"""
之前用过的lasso， 用到了 sparsity 技术，可以用来解决 curse of dimensionality 问题

另一种办法：feature agglomeration， 特征聚集（对feature聚类，注意区别直接对data聚类的阶层式聚类）

正常的聚类
    是根据feature，对data聚类
feature agglomeration
    是数据转置，对feature进行聚类

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, cluster
from sklearn.feature_extraction.image import grid_to_graph

digits = datasets.load_digits()
images = digits.images  # 图片是 (1797, 8, 8)
X = np.reshape(images, (len(images), -1))  # (1797, 64)

connectivity = grid_to_graph(*images[0].shape)  # (64, 64)

"""
开始特征聚类
"""
agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                     n_clusters=32)
agglo.fit(X)

"""
X (1797,64)
缩减到
X_reduced (1797, 32)

FeatureAgglomeration 实例的两个方法要关注：
    1. transform
    2. inverse_transform
"""
X_reduced = agglo.transform(X)  # (1797, 32)
X_approx = agglo.inverse_transform(X_reduced)  # (1797, 64)
images_approx = np.reshape(X_approx, images.shape)  # (1797, 8, 8)

print(images_approx.shape)

IMG_INDEX = 23

plt.figure(1)
print(images[IMG_INDEX])
print(np.unique(images[IMG_INDEX]))
plt.imshow(images[IMG_INDEX])

print('\n===================肉眼看，原图像feature 64的，与压缩图像feature 32的，没什么区别啊')

plt.figure(2)
print(images_approx[IMG_INDEX])
print(np.unique(images_approx[IMG_INDEX]))
plt.imshow(images_approx[IMG_INDEX])

plt.show()
