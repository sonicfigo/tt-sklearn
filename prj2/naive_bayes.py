# coding=utf-8
"""
朴素贝叶斯
这也是著名的机器学习算法，该方法的任务是还原训练样本数据的分布密度，其在多类别分类中有很好的效果。
"""

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from prj2 import l0_data_loader

X, y = l0_data_loader.load()

model = GaussianNB()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
