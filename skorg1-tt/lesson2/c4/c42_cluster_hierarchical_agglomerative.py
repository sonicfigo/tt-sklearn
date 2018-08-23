# coding=utf-8
"""
阶层式的聚类
Hierarchical agglomerative clustering: Ward

阶层式聚类的实现方式众多方法中，有两种是：

agglomerative 优点：数据量大的时候，效率比 k-mean 高
1. Agglomerative - bottom-up approaches:
    each observation starts in its own cluster,
    and clusters are iteratively merged in such a way to minimize a linkage criterion.
    This approach is particularly interesting when the clusters of interest are made of
    only a few observations.
    When the number of clusters is large, it is much more computationally efficient than k-means.

divisive 缺点：数据量大的时候，慢，效率差，从统计学角度来说也是不合适的
2. Divisive - top-down approaches:
    all observations start in one cluster, which is iteratively split as one moves down the hierarchy.
    For estimating large numbers of clusters,
    this approach is both slow (due to all observations starting as one cluster,
    which it splits recursively) and statistically ill-posed.


Connectivity-constrained clustering
With agglomerative clustering, it is possible to specify which samples can be clustered
together by giving a connectivity graph.
Graphs in the scikit are represented by their adjacency matrix.
Often, a sparse matrix is used. This can be useful, for instance,
to retrieve connected regions (sometimes also referred to as connected components) when clustering an image:
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.misc import face

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# #############################################################################
# Generate data
face = face(gray=True)

# Resize it to 10% of the original size to speed up the processing
face = sp.misc.imresize(face, 0.10) / 255.

X = np.reshape(face, (-1, 1))

# #############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*face.shape)  # (7752, 7752)

# #############################################################################

# TODO 还不知道怎么画出tutorial里的face的connected components，先放过
