# coding=utf-8
"""
根据sklearn模型选择图，model优先级：

linear svc
naive bayes

"""
from pprint import pprint

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from datasets import split_movie_review_data

docs_train, docs_test, y_train, y_test, target_names = split_movie_review_data()


def _build_pipeline():
    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    return Pipeline([
        ('vect', TfidfVectorizer(
            # 自己胡乱copy来的参数
            # analyzer='char',  # 这个参数一选上，分数骤降
            # use_idf=False,

            # solution答案里的参数写法，不加的话，分数也不会很低
            min_df=3,
            max_df=0.95
        )),  # TODO 参数何意？
        ('clf', LinearSVC(C=1000))  # TODO 参数何意？
    ])


def build_gscv():
    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    pipe_LinearSVC = _build_pipeline()

    parameters = {'vect__ngram_range': [(1, 1),  # unigrams
                                        (1, 2)]  # bigrams
                  }
    return GridSearchCV(pipe_LinearSVC, parameters, n_jobs=-1)  # TODO 参数何意？


# TASK: print the cross-validated scores for the each parameters set
# explored by the grid search
def _print_gscv(gscv_SGD):
    print('\n===================gscv:')
    print('best_score %s' % gscv_SGD.best_score_)
    # 最佳参数里的ngram_range 为 ngram_range=(1, 2)
    print('bett_estimator %s' % gscv_SGD.best_estimator_)

    print('\n===================gscv-cv_results')
    pprint(gscv_SGD.cv_results_)


# TASK: Predict the outcome on the testing set and store it in a variable
# named y_predicted
def do_predict():
    gscv_SGD = build_gscv()
    gscv_SGD.fit(docs_train, y_train)

    _print_gscv(gscv_SGD)

    y_pred = gscv_SGD.predict(docs_test)
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

    print('\n===================confusion_matrix')
    print(metrics.confusion_matrix(y_test, y_pred))


do_predict()
