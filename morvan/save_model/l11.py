# coding=utf-8
"""
11 Save
我们练习好了一个 model 以后总需要保存和再次预测,
所以保存和读取我们的 sklearn model 也是同样重要的一步.
"""
from __future__ import print_function

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC()
clf.fit(X, y)


# method 1: pickle
def save_pickle():
    import pickle

    # save
    with open('../save_model/clf.pickle', 'w') as f:
        pickle.dump(clf, f)

    # restore
    with open('../save_model/clf.pickle', 'r') as f:
        clf2 = pickle.load(f)
        print(clf2.predict(X[0:1]))


# method 2: joblib
def save_joblib():
    from sklearn.externals import joblib

    # Save
    joblib.dump(clf, '../save_model/clf.joblib')

    # restore
    clf3 = joblib.load('../save_model/clf.joblib')
    print(clf3.predict(X[0:1]))


if __name__ == '__main__':
    save_pickle()
    save_joblib()
