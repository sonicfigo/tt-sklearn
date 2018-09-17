# coding=utf-8
"""
step3.
做token分词后，得到某个单词的全局出现次数
"""
from sklearn.feature_extraction.text import CountVectorizer

from datasets import load_train

twenty_train = load_train()
count_vect = CountVectorizer()  # 派生于 BaseEstimator， 也是一个model


def do_tokenizing():
    # list -> ndarray shape(2257, 35788)
    data_list = twenty_train.data
    X_train_counts = count_vect.fit_transform(data_list)  # (2257, 35788)

    print('\n===================某个单词出现次数')
    vocabulary_dict = count_vect.vocabulary_
    print(vocabulary_dict.get(u'algorithm'))

    return X_train_counts


def get_tokenizer():
    return count_vect


if __name__ == '__main__':
    pass
