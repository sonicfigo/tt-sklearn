# coding=utf-8
"""
Training a machine learning model with scikit-learn:
    http://blog.kaggle.com/2015/04/30/scikit-learn-video-4-model-training-and-prediction-with-k-nearest-neighbors/

问题：
    What is the K-nearest neighbors classification model?
        KNN
        1. 选取一个K
        2. 在unknown点，选择 K 个最近的observations
        3. 在这 K 个里，选择最popular的，作为unknown点的response值

    What are the four steps for model training and prediction in scikit-learn?
        1. 导入model
        2. 实例化一个model
        3. 学习，对model fit数据

    How can I apply this pattern to other machine learning models?
        1. 换一个instance即可，如LogisticRegression()

Resource
    http://scikit-learn.org/stable/modules/neighbors.html
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

# 使用 KNN 对 iris 分类

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X = iris.data
y = iris.target

# print X.shape
# print y.shape

knn = KNeighborsClassifier(n_neighbors=1)  # k换成5试试
knn.fit(X, y)

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

print (knn.predict(X_new))  # 一次predict多个

# 换一个classifier
logreg = LogisticRegression()
logreg.fit(X, y)

print (logreg.predict(X_new))  # 与knn的predict不一样，谁是准的，要validation才可以，见下节
