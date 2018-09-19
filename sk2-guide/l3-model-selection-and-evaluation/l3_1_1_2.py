# coding=utf-8
"""

"""
from sklearn import metrics, svm, datasets
from sklearn.model_selection import cross_val_predict

iris = datasets.load_iris()

clf = svm.SVC(kernel='linear', C=1, random_state=0)
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)

metrics.accuracy_score(iris.target, predicted)
