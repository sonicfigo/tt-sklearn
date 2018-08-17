# coding=utf-8
"""
3.1.1.1. The cross_validate function and multiple metric evaluation

以下两者区别：
cross_validate()
- 多个分数，可以指定多个 evaluation 方法，既scoring可以指定一个list
- 返回的数据较丰富：
    1. train_{ scoring } 分数
    2. test_{ scoring } 分数
    3. fit 耗时
    4. score 耗时

cross_val_score()
- 单个分数
- 传入单个 scoring



"""
from pprint import pprint

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score, cross_validate

iris = datasets.load_iris()
scorings = ['precision_macro', 'recall_macro']

clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores_info = cross_validate(clf, iris.data, iris.target, scoring=scorings,
                             cv=5, return_train_score=True)

pprint(scores_info)

scores_sorted = sorted(scores_info.keys())
print('\n===================排序的key')
print(scores_sorted)

score1 = scores_info['test_recall_macro']
print('\n=================== test_recall_macro 分数')
print(score1)

"""
自定义 scorer
"""

from sklearn.metrics import recall_score
from sklearn.metrics.scorer import make_scorer

scoring_dict = {'prec_macro': 'precision_macro',
                'rec_micro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring_dict,
                        cv=5, return_train_score=True)
