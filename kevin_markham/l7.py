# coding=utf-8
"""
Selecting the best model in scikit-learn using cross-validation

通过cv选择model
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("""
random_state每次不一样，分数都不一样，所以说 单一的testing accuracy 是一个high variance的状态。
改用 cross-validation 解决这种问题。
""")
print(metrics.accuracy_score(y_test, y_pred))


