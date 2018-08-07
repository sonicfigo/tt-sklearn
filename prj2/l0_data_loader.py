# coding=utf-8
"""
UCI大学文件：
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
"""

import numpy as np


def _loadtxt():
    """
    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)
    """
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt('./pima-indians-diabetes.data', delimiter=",")  # 共768行，9列

    # separate the data from the target attributes
    X = dataset[:, 0:7]  # 要所有行， 前7列是features
    y = dataset[:, 8]  # 要所有行， 最后2列是target

    # [   6.     148.      72.      35.       0.      33.6      0.627   50.       1.   ]
    print(dataset[0])

    # [   6.     148.      72.      35.       0.      33.6      0.627]
    print(X[0])

    # 1.0
    print(y[0])

    return X, y


def _normalize(X):
    """
    数据归一化(Data Normalization)

    大多数机器学习算法中的梯度方法对于数据的缩放和尺度都是很敏感的，
    在开始跑算法之前，我们应该进行归一化或者标准化的过程，
    这使得特征数据缩放到0-1范围中。scikit-learn提供了归一化的方法：
    """
    from sklearn import preprocessing
    # normalize the data attributes
    normalized_X = preprocessing.normalize(X)
    # standardize the data attributes
    standardized_X = preprocessing.scale(X)
    return normalized_X, standardized_X


def _feature_selection(X, y):
    """
    特征选择(Feature Selection)

    在解决一个实际问题的过程中，选择合适的特征或者构建特征的能力特别重要。这成为特征选择或者特征工程。
    特征选择时一个很需要创造力的过程，更多的依赖于直觉和专业知识，并且有很多现成的算法来进行特征的选择。
    下面的树算法(Tree algorithms)计算特征的信息量：
    """
    from sklearn import metrics
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(X, y)

    # display the relative importance of each attribute
    print(model.feature_importances_)


def load():
    X, y = _loadtxt()

    # 要不要执行这步处理？ 与不处理X的预测结果比，normal的预测结果差很多, standard还行
    normalized_X, standardized_X = _normalize(X)

    print(normalized_X[0])
    print(standardized_X[0])

    # _feature_selection(X, y)  # 显示哪个feature更重要
    return X, y


if __name__ == '__main__':
    load()
