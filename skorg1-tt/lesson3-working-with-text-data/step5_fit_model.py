# coding=utf-8

"""
step 5，训练一个model-NB，用来做classifier

Now that we have our features,
we can train a classifier to try to predict the category of a post.


"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from datasets import load_train
from step3_tokenizing import get_tokenizer
from step4_tf import do_tfidf_transform, get_tfer

# 要对这2篇文章分类
docs_new = ['God is love',  # soc.religion.christian
            'OpenGL on the GPU is fast']  # comp.graphics

twenty_train = load_train()


def prepare_model():
    X_train_tfidf = do_tfidf_transform()
    y_train = twenty_train.target
    return MultinomialNB().fit(X_train_tfidf, y_train)


clf_NB = prepare_model()


def preprocess_data():
    tokenizer = get_tokenizer()  # 辅助1
    tfer = get_tfer()  # 辅助2

    """
    Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task.

    scikit-learn includes several variants of this classifier;
    the one most suitable for word counts is the multinomial variant:
    """

    # 同理两个步骤
    # 1.分词 transform，转换成ndarray。不要fit，老已fit过
    X_new_counts = tokenizer.transform(docs_new)
    # 2. TF transform, 计算TF，同样不要fit
    X_new_tfidf = tfer.transform(X_new_counts)
    return X_new_tfidf


X_new_tfidf = preprocess_data()

predicted = clf_NB.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
