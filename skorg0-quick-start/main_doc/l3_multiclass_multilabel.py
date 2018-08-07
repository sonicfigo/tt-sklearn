# coding=utf-8
"""
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#multiclass-vs-multilabel-fitting

target，进行二值化，既从明文，转为值为 0 or 1 的矩阵
"""

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]  # features
y = [0, 0, 1, 1, 2]  # target 随意修改，反正对应feature

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print('\n原始预测')
print(classif.fit(X, y).predict(X))
# array([0, 0, 1, 1, 2])


y = LabelBinarizer().fit_transform(y)
print('\ny 转为2D后：')
print(y)

print('\ntransform后的预测结果：')
print(classif.fit(X, y).predict(X))

# array([[1, 0, 0],
#        [1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 0],
#        [0, 0, 0]])
