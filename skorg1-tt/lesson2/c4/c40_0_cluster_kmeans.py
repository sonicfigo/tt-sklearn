# coding=utf-8
"""
前提：
    - 给定 iris 数据
    - 已知有 3 个分类
操作：
    把这些数据聚类，每个标注好3分类标签中的其一(0 or 1 or 2 类）
"""

from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

k_means = KMeans(n_clusters=3)
k_means.fit(X_iris)

print("""
聚3类的结果，前十个
vs
标准答案，前十个
（原始数据是三类规整排序好的，所以学习后的聚类结果，也是三类排序好的）""")
print(k_means.labels_[::10])  # 从头开始，每次跳10个
print(y_iris[::10])

print("""
完整聚类结果
vs
完整标准答案""")
print(k_means.labels_)
print(y_iris)
