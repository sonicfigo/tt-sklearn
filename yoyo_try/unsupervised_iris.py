# coding=utf-8
"""
用非监督，看iris分类结果是不是0,1,2
还是需要人为的强制分为3类，只看是否三类的边界，是否与target一致或差不多
"""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()

kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(iris.data)
print (kmeans.labels_)
print (iris.target)
