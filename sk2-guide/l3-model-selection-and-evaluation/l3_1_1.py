# coding=utf-8
"""
3.1.1. Computing cross-validated metrics

"""
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score, ShuffleSplit

iris = datasets.load_iris()

clf = svm.SVC(kernel='linear', C=1)

"""
直接用 cross_val_score
"""
scores1 = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

"""
使用一个 
cross validation iterator

既是 ShuffleSplit
"""

n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores2 = cross_val_score(clf, iris.data, iris.target, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

"""
Data transformation with held out data

预处理：
学习时，要作用在 train 的 X
考试时，也要先作用在 test 的 X
"""

from sklearn import preprocessing, datasets, svm
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)

clf = svm.SVC(C=1).fit(X_train_transformed, y_train)  # 学习的是转换过的X

X_test_transformed = scaler.transform(X_test)  # 考试的X也要转换下

"""
pipeline 可以把这个标准化的动作，整合到 cross_val_score

A Pipeline makes it easier to compose estimators, 
providing this behavior under cross-validation
"""
from sklearn.pipeline import make_pipeline

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
clf2 = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

cross_val_score(clf2, iris.data, iris.target, cv=cv)
