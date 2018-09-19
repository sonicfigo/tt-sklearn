# coding=utf-8
"""
打破一切，一次性用pipeline做完所有步骤：

vectorizer => transformer => classifier

SVM 比 NB 更适合做 text 分类， 通过 混淆矩阵 和 classification 报告，可以观察到
"""
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from datasets import load_train_4kinds_newsgp, load_test_4kinds_newsgp
from sklearn.pipeline import Pipeline

twenty_train = load_train_4kinds_newsgp()
train_data_docs = twenty_train.data  # list
y_train = twenty_train.target  # (2257, )

twenty_test = load_test_4kinds_newsgp()
test_data_docs = twenty_test.data


def _build_model_NB():
    pipe_NB = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    pipe_NB.fit(train_data_docs, y_train)  # fit原始数据, <list> 即可
    return pipe_NB


def _build_model_SGD():
    from sklearn.linear_model import SGDClassifier  # SVM , logistic regression
    pipe_SGD = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None)),
                         ])
    pipe_SGD.fit(train_data_docs, y_train)
    return pipe_SGD


pipe_NB = _build_model_NB()
pipe_SGD = _build_model_SGD()


def do_predict():
    # 要对这2篇文章分类
    docs_new = ['God is love',  # soc.religion.christian
                'OpenGL on the GPU is fast']  # comp.graphics
    predicted = pipe_NB.predict(docs_new)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))


def evaluation_NB():
    """评价这个 NB 的分数如何"""

    predicted = pipe_NB.predict(test_data_docs)

    print(metrics.classification_report(twenty_test.target, predicted,
                                        target_names=twenty_test.target_names))
    print(metrics.confusion_matrix(twenty_test.target, predicted))

    return np.mean(predicted == twenty_test.target)  # 0.834886817577


def evaluation_SGD():
    """SVM 分数更高"""

    predicted = pipe_SGD.predict(test_data_docs)

    print(metrics.classification_report(twenty_test.target, predicted,
                                        target_names=twenty_test.target_names))
    print(metrics.confusion_matrix(twenty_test.target, predicted))

    return np.mean(predicted == twenty_test.target)  # 0.912782956059


if __name__ == '__main__':
    # do_predict()
    print(evaluation_NB())
    # print(evaluation_SGD())


"""
SGD 结果

precision：  准确率，既某个类别，正确识别的数 / 所有识别为该类的数
            例子：第一类 258 / (258+4+5+5) = 0.95
            
recall：     召回率，就是正确识别的数 / 该label总数(support)
             例子：第一类，258 / 319 = 0.81


------------------------------------------ classification_report
                        precision    recall  f1-score   support

           alt.atheism       0.95      0.81      0.87       319
         comp.graphics       0.88      0.97      0.92       389
               sci.med       0.94      0.90      0.92       396
soc.religion.christian       0.90      0.95      0.93       398

           avg / total       0.92      0.91      0.91      1502


------------------------------------------      confusion_matrix
[[258  11  15  35]
 [  4 379   3   3]
 [  5  33 355   3]
 [  5  10   4 379]]


既是：
                            alt.atheism  comp.graphics  sci.med      oc.religion.christian
           alt.atheism  [   258         11              15           35                  ]
         comp.graphics  [     4         379             3            3                   ]
               sci.med  [     5         33              355          3                   ]
soc.religion.christian  [     5         10              4            379                 ]


"""
