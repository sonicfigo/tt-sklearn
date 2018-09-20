# coding=utf-8
"""
一般在文本特征处理过程中，本来正常的流程就是先用CountVectorizer来提取特征，
然后就用TfidfTransformer来计算特征的权重。

而TfidfVectorizer则是把两者的功能合在一起，连参数也都是两者的参数合在一起，
所以可以方便的直接使用TfidfVectorizer。

但是如果想在CountVectorizer来提取特征后想处理特征，比如降维之类的，
这样直接使用TfidfVectorizer就不行了。

"""
from sklearn.feature_extraction.text import TfidfVectorizer

# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
                             use_idf=False)


def tfidf_vectorizer():
    data_list = ['aab',
                 'ab',
                 'cd']

    X_train_tfidf = vectorizer.fit_transform(data_list)

    print(X_train_tfidf)
    print(type(X_train_tfidf))
    print(X_train_tfidf.shape)


tfidf_vectorizer()
