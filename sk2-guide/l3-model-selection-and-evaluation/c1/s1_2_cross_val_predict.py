# coding=utf-8
"""
都是在做 cv，输出结果
cross_val_score：    单个分数
cross_validate:     多个分数
cross_val_predict:  pred预测值
"""

from sklearn import metrics, svm, datasets
from sklearn.model_selection import cross_val_predict

iris = datasets.load_iris()

clf = svm.SVC(kernel='linear', C=1, random_state=0)
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)

"""
看下 target-真实 vs predicted-预测
"""
print(metrics.accuracy_score(iris.target, predicted))
