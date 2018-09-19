# coding=utf-8
"""
step3. CountVectorizer 用来做token分词后，得到某个单词的全局出现次数
"""
from sklearn.feature_extraction.text import CountVectorizer

from datasets import load_train_4kinds_newsgp

twenty_train = load_train_4kinds_newsgp()
count_vect = CountVectorizer()  # 派生于 BaseEstimator， 也是一个model


def do_tokenizing():
    # list -> ndarray shape(2257, 35788)
    data_list = twenty_train.data

    # <class 'scipy.sparse.csr.csr_matrix'> ，此类也有shape，(2257, 35788)
    X_train_tokenized = count_vect.fit_transform(data_list)

    print('\n===================fit后，就有某个单词的出现次数')
    vocabulary_dict = count_vect.vocabulary_
    print(vocabulary_dict.get(u'algorithm'))  # int , 如 4690

    # 每个单词出现的次数
    return X_train_tokenized


def get_tokenizer():
    return count_vect


if __name__ == '__main__':
    do_tokenizing()
