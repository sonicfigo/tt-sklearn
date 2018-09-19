# coding=utf-8
"""

"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier  # SVM , logistic regression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from datasets import load_train_4kinds_newsgp

twenty_train = load_train_4kinds_newsgp()
docs_train = twenty_train.data  # list
y_train = twenty_train.target  # (2257, )

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }

pipe_SGD = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
                     ])
pipe_SGD.fit(docs_train, y_train)

gscv_SGD = GridSearchCV(pipe_SGD, parameters, n_jobs=-1)
gscv_SGD = gscv_SGD.fit(docs_train[:400], y_train[:400])

foo_doc_type = twenty_train.target_names[gscv_SGD.predict(['God is love'])[0]]
print(foo_doc_type)  # soc.religion.christian

print(gscv_SGD.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gscv_SGD.best_params_[param_name]))

"""
clf__alpha: 0.001
tfidf__use_idf: True
vect__ngram_range: (1, 1)
"""

print('\n===================更详细的结果 cv_results_')
print(gscv_SGD.cv_results_)
