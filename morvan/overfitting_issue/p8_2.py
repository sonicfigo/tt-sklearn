# coding=utf-8
"""
cross validation 交叉验证1

------------8.2 cross 方法， 综合了五个分组，比较没有偏差
前提： n_neighbors = 5
"""
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()
X = iris.data
y = iris.target

# this is cross_val_score #
knn = KNeighborsClassifier(n_neighbors=5)

# cross_val_score 包装隐藏了 train_test_split 步骤吗？？？？？
# cv5，进行五次分组
# scoring 表示如何判断分数，用的是accuracy方式，准确性
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

print('\n===================list of 五组分数')
print(scores)

print('\n===================平均分数(这个model，+这个参数，分数是如下)')
print(scores.mean())  # scores / 5
