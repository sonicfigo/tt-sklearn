# coding=utf-8

"""
=========================================================
Vector Quantization Example
=========================================================

Face, a 1024 x 768 size image of a raccoon face,
is used here to illustrate how `k`-means is
used for vector quantization.

#### 原图
    - shape:  (768, 1024)
    - unique: 251个，既 0 ~ 250
#### 压缩
    compressed face
        - model: kmeans
        - 最终结果
            - shape: (768, 1024)
            - unique：[  27.62031146   75.41095451  114.99362851  153.31393344  194.13840989]
vs

    bin face
        - model: 无，使用 linspace, searchsorted, choose 变相缩减了
        - 最终结果
            - shape: (768, 1024)
            - unique: [  25.6   76.8  128.   179.2  230.4]
"""

import numpy as np
from scipy.misc import face
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# FIG_SIZE = (3, 2.2)
FIG_SIZE = (13, 7)

face = face(gray=True)  # (768, 1024)

n_clusters = 5
np.random.seed(1)

# (768, 1024) -> (786432, 1)
X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
model_k_means = KMeans(n_clusters=n_clusters, n_init=4)
model_k_means.fit(X)
# cluster_centers_ 既是 kmean 出来的 5 个 centroids
centroid_values_5 = model_k_means.cluster_centers_.squeeze()

vmin = face.min()
vmax = face.max()


# original face
def plot_origin_face():
    plt.figure(1, figsize=FIG_SIZE)
    plt.imshow(face, cmap=plt.cm.gray, vmin=vmin, vmax=256)


# compressed face
def plot_compressed_face():
    """
    kemans , n=5，已经处理了 786432 个原始数据，分为5类label了
    这786432 个 5类label值，对着分类后的5个中心点的值，执行choose， 变成 786432 个中心点值
    """
    labels_786432 = model_k_means.labels_
    # create an array from labels and values
    face_compressed = np.choose(labels_786432, centroid_values_5)
    face_compressed.shape = face.shape

    plt.figure(2, figsize=FIG_SIZE)
    plt.imshow(face_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    print('\n===================face_compressed')
    print(face_compressed.shape)


def plot_bins_face():
    """
    equal bins face
    跟原图很接近，除非放大仔细看，细节处可以看出像素点的粗糙

    缩减：
        - 原来的face，unique 值：0 ~ 250， 250个
        - 现在的face，unique 值 [  25.6   76.8  128.   179.2  230.4]， 5个
    """

    """
    0 ~ 256(应该是256色的意思)，均分成 5段，6个点
    [   0.    51.2  102.4  153.6  204.8  256. ]
    """
    regular_values_6of256 = np.linspace(0, 256, n_clusters + 1)

    """
    searchsorted，face 每点的值，处于 regular_6values 的哪个index位置
    这样face 的点，都变成 0 ~ 6 这7个值其中一个，就可以作为 labels 了
    
    [[2 2 2 ..., 2 2 2]
     [1 2 2 ..., 2 2 2]
     [1 1 2 ..., 2 2 2]
     ..., 
     [1 2 2 ..., 2 2 2]
     [1 2 2 ..., 2 2 2]
     [1 2 2 ..., 2 2 2]]
    """
    regular_labels_768_1024 = np.searchsorted(regular_values_6of256,
                                              face) - 1  # (768, 1024)

    """
    mean：[  25.6   76.8  128.   179.2  230.4]
    """
    regular_values_6of256 = .5 * (regular_values_6of256[1:] + regular_values_6of256[:-1])

    """
    ravel 打平成 786432 个 标签值
    choose，又把 标签值，转成了 数值，真正用来imshow用
    """
    regular_face_786432 = np.choose(regular_labels_768_1024.ravel(),
                                    choices=regular_values_6of256, mode="clip")
    regular_face_786432.shape = face.shape  # (786432,) -> (768, 1024)
    regular_face_786_1024 = regular_face_786432

    plt.figure(3, figsize=FIG_SIZE)
    plt.imshow(regular_face_786_1024, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

    print('\n===================regular_face')
    print(regular_face_786_1024.shape)
    return regular_values_6of256


def plot_histogram(regular_values):
    """不知道干嘛的"""
    plt.figure(4, figsize=FIG_SIZE)
    plt.clf()
    plt.axes([.01, .01, .98, .98])
    plt.hist(X, bins=256, color='.5', edgecolor='.5')
    plt.yticks(())
    plt.xticks(regular_values)
    values_sorted = np.sort(centroid_values_5)
    for center_1, center_2 in zip(values_sorted[:-1], values_sorted[1:]):
        plt.axvline(.5 * (center_1 + center_2), color='b')

    for center_1, center_2 in zip(regular_values[:-1], regular_values[1:]):
        plt.axvline(.5 * (center_1 + center_2), color='b', linestyle='--')


plot_origin_face()
plot_compressed_face()
regular_face = plot_bins_face()
plot_histogram(regular_face)

plt.show()
