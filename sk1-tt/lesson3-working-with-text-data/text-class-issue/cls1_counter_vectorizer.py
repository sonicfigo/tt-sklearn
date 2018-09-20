# coding=utf-8
"""
sklearn中一般使用
- CountVectorizer
- TfidfVectorizer
这两个类来提取文本特征


理解 CountVectorizer 的基础用法

把单词在每个doc出现的次数，转成一个 vector

        word0   word1   word2   word3   word4   word5
doc0
doc1
doc2

"""
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()


def count_vectorizer_issue():
    data_list = ['foo bar foo',
                 'england a usa usa england usa bar bar bar bar bar',
                 'baz foo baz']

    X_train_count_vectorized = count_vect.fit_transform(data_list)
    """
         
          (doc索引，文字id ) -> 在本doc出现次数

          稀疏矩阵表达方式
          (0, 0 bar)        ->1
          (0, 4 foo)        ->2

          (1, 0 bar)        ->5
          (1, 2 england)    ->1
          (1, 3 china)      ->2
          (1, 5 usa)        ->3

          (2, 1 baz)        ->2
          (2, 4 foo)        ->1

        shape 是 （3，6）
        3行： 3个doc
        6列： 6个word
        """
    print(X_train_count_vectorized)

    """
    to array ，可以类似观察完整矩阵
        [[1 0 0 0 2 0]
        [5 0 1 2 0 3]
        [0 2 0 0 1 0]]
    """
    print(X_train_count_vectorized.toarray())

    print('\n=================== get_feature_names')
    # [u'bar', u'baz', u'china', u'england', u'foo', u'usa']
    print(count_vect.get_feature_names())

    print('\n=================== scipy.sparse.csr.csr_matrix')
    print(type(X_train_count_vectorized))

    print('\n===================shape(3, 6)')
    print(X_train_count_vectorized.shape)

    print('\n===================文字: 对应的id')
    print(count_vect.vocabulary_)


count_vectorizer_issue()
