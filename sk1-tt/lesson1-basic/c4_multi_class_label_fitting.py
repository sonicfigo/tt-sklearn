# coding=utf-8
"""
fit 的 y
与
predict 的 y

fit 一维，predict出来是一维
fit 二维，predict出来是二维

"""

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
print("""
原始y %s""" % y)

print("""
1.fit by:     1维multiclass label
predict:    互斥多元分类结果(1d multiclass)""")
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(X, y).predict(X))  # 输出也是一维多元

# Here, the classifier is fit() on a 2d binary label representation of y, using the LabelBinarizer.

y = LabelBinarizer().fit_transform(y)  # shape(5,3)
print("""
2.fit by:     2维binary label
predict:    2维非互斥多元分类结果(2d multilabel)
变形后的y
%s""" % y)
print("""predict：
最后两行结果为 000，表示predict结果，不在原来y(0,1,2)三种范围内。""")
print(classif.fit(X, y).predict(X))
