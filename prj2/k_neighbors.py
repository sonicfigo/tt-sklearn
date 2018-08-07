# coding=utf-8
"""
k近邻
k近邻算法常常被用作是分类算法一部分，比如可以用它来评估特征，在特征选择上我们可以用到它。
"""

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from prj2 import l0_data_loader

X, y = l0_data_loader.load()

# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
