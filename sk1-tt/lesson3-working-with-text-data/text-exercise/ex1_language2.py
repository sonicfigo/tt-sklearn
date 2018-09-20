# coding=utf-8
"""
使用模型：
  TfidfVectorizer + Perceptron

分数与 sgd + gscv 的相差不大
"""

from datasets import split_paragraphs_data
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline

docs_train, docs_test, y_train, y_test, target_names = split_paragraphs_data()

vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
                             use_idf=False)


def _build_pipeline():
    return Pipeline([('vect', vectorizer),
                     ('clf', Perceptron(tol=1e-3)),
                     ])


def predict_and_verify_performance():
    pipe1 = _build_pipeline()

    pipe1.fit(docs_train, y_train)
    y_pred = pipe1.predict(docs_test)

    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    print(metrics.confusion_matrix(y_test, y_pred))
    return pipe1


pipe1 = predict_and_verify_performance()


def back_into_real_world():
    print('\n===================拿一些句子来做测试')
    sentences = [
        u'This is a language detection test',
        u'Ceci est un test de d\xe9tection de la langue.',
        u'Dies ist ein Test, um die Sprache zu erkennen.',
    ]
    predicted = pipe1.predict(sentences)
    for s, p in zip(sentences, predicted):
        print(u'The language of "%s" is "%s"' % (s, target_names[p]))


back_into_real_world()
