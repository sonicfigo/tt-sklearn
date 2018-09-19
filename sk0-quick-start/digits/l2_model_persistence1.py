# coding=utf-8
"""
model save，略过
"""

import pickle

from sklearn import svm, datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# 存储到StringIO
# str1 = pickle.dumps(clf)
# clf2 = pickle.loads(str1)
# y = clf2.predict(X[0:1])
# print y[0]

from sklearn.externals import joblib

joblib.dump(clf, 'filename.pkl')
