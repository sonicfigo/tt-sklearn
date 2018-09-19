# coding=utf-8
"""
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#multiclass-vs-multilabel-fitting

target，进行二值化，既从明文，转为值为 0 or 1 的矩阵

例子中，是假设正在做一个多类别分类
"""

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier  # 随意用的，无所谓，这个类确切用法是用来进行多分类的
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]  # features
y = [0, 0, 1, 1, 2]  # target 随意设置，表示一个分类结果，例子中既有3类，0，1，2

classif = OneVsRestClassifier(estimator=SVC(random_state=0))

print('\n原 predict，predict结果为 1d，表示类别 0，1，2')
y_predict = classif.fit(X, y).predict(X)
print(y_predict.shape)
print(y_predict)
# array([0, 0, 1, 1, 2])


y = LabelBinarizer().fit_transform(y)  # 用这个 binarizer 二值化
print('\ny 转为2D后：')
print(y)

print('\ntransform后的predict结果为 2d， 每一行表示对应的类别，如 [1,0,0] 表示类别 0')
y_predict = classif.fit(X, y).predict(X)
print(y_predict.shape)
print(y_predict)

# array([[1, 0, 0],
#        [1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 0],
#        [0, 0, 0]])
