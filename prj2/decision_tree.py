# coding=utf-8
"""
决策树
分类与回归树(Classification and Regression Trees ,CART)算法常用于特征含有类别信息的分类或者回归问题，这种方法非常适用于多分类情况
"""

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from prj2 import l0_data_loader

X, y = l0_data_loader.load()

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
