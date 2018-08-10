# coding=utf-8
"""
normalization ，又叫scale

scale与否，影响考试分数
"""
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC  # 支持向量机 的 支持向量classifier

X, y = make_classification(n_samples=300,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           random_state=22,  # random_state=22 确定每次随机生成数据都一样
                           n_clusters_per_class=1,
                           scale=100)  # X shape (300, 2), y shape (300, )

"""
不scale X的学习并考试 低分
"""
print('\n===================显示原始X数据的图形(横轴，竖轴的值都很大(-300 ~ 400, -200 ~ 400))')
print(X[0])
plt.scatter(X[:, 0], X[:, 1], c=y)  # 横轴：X的column1， 竖轴：X的column2
plt.show()

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=.3)
clf1 = SVC()
clf1.fit(X_train1, y_train1)
print('\n原始X数据不 scale，考试低分，大概0.5分')
print (clf1.score(X_test1, y_test1))

"""
scale后(两个轴的范围都缩小了，看plt图形)，学习并考试 高分
"""
X2 = preprocessing.scale(X)
# preprocessing.scale 既等于
# preprocessing.minmax_scale(X, feature_range=(0, 1)), 浓缩到0 ~ 1之间

plt.scatter(X2[:, 0], X2[:, 1], c=y)
plt.show()

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=.3)
clf2 = SVC()
clf2.fit(X_train2, y_train2)  # 学习
print('\n===================scale 后，x轴y轴都浓缩范围了，考试也高分了，大概0.9分')
print (clf2.score(X_test2, y_test2))
