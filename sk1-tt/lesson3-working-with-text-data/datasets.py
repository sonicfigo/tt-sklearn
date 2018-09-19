# coding=utf-8
"""
取数据工具类
"""
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.model_selection import train_test_split

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


def load_train_4kinds_newsgp():
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories, shuffle=True,
                                      random_state=42)
    return twenty_train


def load_test_4kinds_newsgp():
    twenty_test = fetch_20newsgroups(subset='test',
                                     categories=categories, shuffle=True,
                                     random_state=42)
    return twenty_test


paragraphs_dir = '/Users/figo/pcharm/ml/tt-sklearn/skorg1-tt/lesson3-working-with-text-data/paragraphs'


def load_train_and_test_paragraphs():
    lang_paragraphs = load_files(paragraphs_dir)
    docs_all = lang_paragraphs.data
    y = lang_paragraphs.target
    print lang_paragraphs.target_names
    docs_train, docs_test, y_train, y_test = train_test_split(docs_all, y, test_size=0.8)
    return docs_train, docs_test, y_train, y_test, lang_paragraphs.target_names


if __name__ == '__main__':
    load_train_and_test_paragraphs()
