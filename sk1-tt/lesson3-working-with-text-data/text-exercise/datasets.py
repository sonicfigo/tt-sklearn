# coding=utf-8
"""
取数据工具类
"""
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

paragraphs_dir = '/Users/figo/pcharm/ml/tt-sklearn/sk1-tt/lesson3-working-with-text-data/text-exercise/datas/paragraphs'
movie_review_dir = '/Users/figo/pcharm/ml/tt-sklearn/sk1-tt/lesson3-working-with-text-data/text-exercise/datas/txt_sentoken'


def split_paragraphs_data():
    lang_paragraphs = load_files(paragraphs_dir)

    docs_all = lang_paragraphs.data
    y = lang_paragraphs.target

    docs_train, docs_test, y_train, y_test = train_test_split(docs_all, y, test_size=0.5)
    return docs_train, docs_test, y_train, y_test, lang_paragraphs.target_names


def split_movie_review_data():
    reviews = load_files(movie_review_dir)

    docs_all = reviews.data
    y = reviews.target

    docs_train, docs_test, y_train, y_test = train_test_split(docs_all, y, test_size=0.25)
    return docs_train, docs_test, y_train, y_test, reviews.target_names


if __name__ == '__main__':
    # split_paragraphs_data()
    split_movie_review_data()
