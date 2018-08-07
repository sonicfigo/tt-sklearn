# coding=utf-8
"""
逻辑回归
大多数问题都可以归结为二元分类问题。这个算法的优点是可以给出数据所在类别的概率。
"""

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from prj2 import l0_data_loader

X, y = l0_data_loader.load()

model = LogisticRegression()
model.fit(X, y)
print(model)

# make predictions
expected = y
predicted = model.predict(X)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
