# coding=utf-8
"""

"""

from sklearn.feature_extraction.text import CountVectorizer

X_test = ['没有 你 的 地方 都是 他乡', '没有 你 的 旅行 都是 流浪']

count_vec = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b")

print count_vec.fit_transform(X_test).toarray()

print count_vec.fit_transform(X_test)

print '\nvocabulary list:\n'
for key, value in count_vec.vocabulary_.items():
    print key, value

print
