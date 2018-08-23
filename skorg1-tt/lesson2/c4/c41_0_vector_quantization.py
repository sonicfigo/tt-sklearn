# coding=utf-8
"""
Application example: vector quantization

vector quantization
矢量量化（VQ，Vector Quantization）是一种极其重要的信号压缩方法。

VectorQuantization (VQ)是一种基于块编码规则的有损数据压缩方法。
事实上，在 JPEG 和 MPEG-4 等多媒体压缩格式里都有 VQ 这一步。
它的基本思想是：
    将若干个标量数据组构成一个矢量，然后在矢量空间给以整体量化，从而压缩了数据而不损失多少信息。

PS：
标量：       无方向
矢量/向量：  有方向，在物理学中称作矢量，在数学中称作向量。


几个主要对象的shape
    值的范围很大
        face                :   (768, 1024)
        X                   :   (786432, 1)

    值的范围只有 0 ~ 4
        labels_             :   (786432,)
        values              :   (5,)
        face_comporessed    :   (768, 1024)

总结，cluster.KMeans 作用，得到两个属性：
    - labels_
        就是把 786432 个点，值范围，从原来的N类（N未知，反正远大于5）的 ，缩减成只有5类 (仅仅是 5 类的index, 0,1,2,3,4)
    - cluster_centers_
        5个类的具体值，也叫码矢，质心

    最后用np.choose函数 ，参数：index 是 786432 值， choices是 cluster_centers_，得到缩减后的数据，shape 不变，distinct 值较小为5


"""
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

try:
    face = sp.face(gray=True)

except AttributeError:
    from scipy import misc

    face = misc.face(gray=True)

# (768, 1024) -> (786432, 1)
X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array

"""
n_clusters = 5
所有颜色点，缩减为 5 类
所以 values.shape = (5, )
"""
k_means = KMeans(n_clusters=5, n_init=1)
k_means.fit(X)

"""
labels_of_786432 
就是原图片， 的786432个图像点，通过n  kmeans 缩减后
原来值范围：不确定，但一定是>5种，如 [114 130 145 ..., 119 129 137]
变为
现在值范围：0 ~ 4 之间了，如  [2 1 1 ..., 1 1 1]
"""
labels_of_786432 = k_means.labels_
print('\n=================== labels\n %s' % labels_of_786432)  # (786432,)

"""
cluster_centers_ 既是 kmean 出来的 5 个 centroids
真实数据值 [ 191.63450508  109.71975211   71.49771602  148.25700663   26.11714144]

squeeze：只是把shape从 (5, 1) -> (5, )
"""
values_of_5_centroids = k_means.cluster_centers_.squeeze()
print('\n=================== values\n %s' % values_of_5_centroids)

print('\n===================按 labels 的index，来取values')
face_compressed = np.choose(labels_of_786432, values_of_5_centroids)
face_compressed.shape = face.shape  # (768, 1024)

print('\n===================原始对象')
print(face)
print(face_compressed)

print(k_means.cluster_centers_.squeeze())
