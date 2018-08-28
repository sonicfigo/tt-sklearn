# coding=utf-8
"""
非官网，网络帖子里的书写例子
除了最后一个学习器之外，前面的所有学习器必须提供transform方法，
该方法用于数据转化，如：
    - 归一化
    - 正则化
    - 特征提取

若没有，就异常
"""

from sklearn.datasets import load_digits
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

m2 = LogisticRegression(C=1)


def test_Pipeline_ex(data):
    m1 = LinearSVC(C=1, penalty='l1', dual=False)

    pipeline = Pipeline(steps=[('Linear_SVM', m1),
                               ('LogisticRegression', m2)])
    x_train, x_test, y_train, y_test = data
    pipeline.fit(x_train, y_train)
    print('name steps:', pipeline.named_steps)
    print('Pipeline Score:', pipeline.score(x_test, y_test))


"""
工作流程：先进行pca降为，然后使用Logistic回归，来分类
"""


def test_Pipeline_ok(data):
    pipeline = Pipeline(steps=[('PCA', PCA()),
                               ('LogisticRegression', m2)])

    x_train, x_test, y_train, y_test = data
    pipeline.fit(x_train, y_train)

    print('name steps:', pipeline.named_steps)
    print('Pipeline Score:', pipeline.score(x_test, y_test))


if __name__ == '__main__':
    data = load_digits()
    X = data.data
    y = data.target
    try:
        test_Pipeline_ex(train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=y))
    except BaseException as ex:
        print('\n===================error:')
        print(ex)

    print('\n===================ok:')
    test_Pipeline_ok(train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y))
