# coding=utf-8
"""
练习一：各国语言的文本
1. 训练
2. 识别出语言
"""

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from datasets import load_train_and_test_paragraphs

docs_train, docs_test, y_train, y_test, target_names = load_train_and_test_paragraphs()


def _build_pipeline():
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
                     ])


def build_gscv():
    pipe_SGD = _build_pipeline()
    pipe_SGD.fit(docs_train, y_train)

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
    return GridSearchCV(pipe_SGD, parameters, n_jobs=-1)


def predict_and_verify_performance():
    gscv_SGD = build_gscv()

    gscv_SGD.fit(docs_train, y_train)
    y_pred = gscv_SGD.predict(docs_test)

    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    print(metrics.confusion_matrix(y_test, y_pred))
    return gscv_SGD


gscv_SGD = predict_and_verify_performance()

print('\n===================拿一些句子来做测试')


def back_into_real_world():
    sentences = [
        u'This is a language detection test',
        u'Ceci est un test de d\xe9tection de la langue.',
        u'Dies ist ein Test, um die Sprache zu erkennen.',
    ]
    predicted = gscv_SGD.predict(sentences)
    for s, p in zip(sentences, predicted):
        print(u'The language of "%s" is "%s"' % (s, target_names[p]))


back_into_real_world()
