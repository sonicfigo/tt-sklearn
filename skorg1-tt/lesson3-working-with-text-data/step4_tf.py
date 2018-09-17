# coding=utf-8
"""
step4. TF

occurences：出现次数，并不能完全精准表达一个doc的主题。因为如果doc长度长，那么occurrences 本身自然就会高。
所以改用 tf - term frequencies 这个概念来衡量
"""
from sklearn.feature_extraction.text import TfidfTransformer

from step3_tokenizing import do_tokenizing

X_train_counts = do_tokenizing()  # (2257, 35788)


def _do_tf_transform():
    """
    演示而已，没人调用
    两个动作：
    1. fit：喂数据
    2. transform：转换，our count-matrix =========> a tf-idf representation.
    """
    # TfidfTransformer 类，派生于 BaseEstimator， 也是一个estimator
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)  # (2257, 35788)
    return X_train_tf


tfidf_transformer = TfidfTransformer()  # default use_idf=True


def do_tfidf_transform():
    """
    真正被调用到的
    两个动作2合1：
    fit_transform(..)

    use_idf = True，适用于文章很短的情况
    """

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)  # (2257, 35788)
    return X_train_tfidf


def get_tfer():
    return tfidf_transformer


if __name__ == '__main__':
    _do_tf_transform()
