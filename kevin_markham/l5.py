# coding=utf-8
"""
Comparing machine learning models in scikit-learn
    http://blog.kaggle.com/2015/05/14/scikit-learn-video-5-choosing-a-machine-learning-model/

问题:
    How do I choose which model to use for my supervised learning task? 选算法

    How do I choose the best tuning parameters for that model? 选参数

    How do I estimate the likely performance of my model on out-of-sample data?

方案：
    Model evaluation，有两种procedure:
    1. train 和 test 同一套数据。不好
    2. train test split 。好 （cross_val_score的低端版，cross_val_score更好）

"""
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# 第一种procedure，不好
logreg = LogisticRegression()
logreg.fit(X, y)
print (metrics.accuracy_score(y_true=y, y_pred=logreg.predict(X)))

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X, y)
print (metrics.accuracy_score(y_true=y, y_pred=knn5.predict(X)))

# k=1时，fit和predict若是同一套数据，那么accuracy永远是100%，因为：
# 在当前点(predict的数据点)，找附近的最近的一个点(fit的数据点)，那就是自己（因为是同一套数据）。
# 所以可以得出一个结论：fit和predict是同一套数据时，不适合用来validation 一个model的好坏。
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X, y)
print (metrics.accuracy_score(y_true=y, y_pred=knn1.predict(X)))  # best

# ======================第二种procedure，好
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)  # 20% 或40%常见
knn0 = KNeighborsClassifier(n_neighbors=1)
knn0.fit(X_train, y_train)

# 准确率会随着随机数据的变化，而变化，除非使用 random_state=n
print (metrics.accuracy_score(y_true=y_test, y_pred=knn0.predict(X_test)))
# 上一句code等同于    print knn0.score(X_test, y_test)

print ('shape:')
print (X_train.shape)
print (X_test.shape)
