# coding=utf-8
"""
Choosing the parameters of the model

fit函数 传入 training set，完成学习.
"""

from sklearn import datasets, svm

digits = datasets.load_digits()


def _create_classifier():
    """
    创建一个分类器，并填充训练集-fit，进行学习。
    """
    # In this example we set the value of gamma manually.
    # It is possible to automatically find good values for the parameters by using tools
    # such as grid search and cross validation.
    clf = svm.SVC(gamma=0.001, C=100.)
    print ('\n开始学习：fit by traning set:    %s  (%s个).' %
           (digits.target, len(digits.target)))
    clf.fit(X=digits.data, y=digits.target)  # how many to learn
    return clf


classifier = _create_classifier()

print ('\n完成学习后，才可以预测结果：(predict(预测), actual(答案))')
predict_digits = classifier.predict(digits.data[-10:])
print(zip(predict_digits, digits.target[-10:]))
