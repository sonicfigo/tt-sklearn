# coding=utf-8
"""
线性回归
Data science in Python: pandas, seaborn, scikit-learn
    How do I use the pandas library to read data into Python?
    How do I use the seaborn library to visualize data?

    What is linear regression, and how does it work?

    How do I train and interpret a linear regression model in scikit-learn?

    What are some evaluation metrics for regression problems?
    How do I choose which features to include in my model?

数据的意义
    已有：200个market里的，TV, Radio, Newpaper 三个广告费的转化率
    预测：转化率

目的：
    使用已有数据，建一个linear regression model，用来预测新数据

评估-metrics：
    MAE 最简单
    MSE
    RMSE root MSE, MSE的开根

    RMSE最好

选择feature：
    Newspaper 确实belong 我们的model吗？目前可以做两个步骤验证：
    1. 看fit后的coefficient。
    2. 去掉这个feature后，predict的RMSE成功率是不是高了。
"""
import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data():
    df = pd.read_csv(
        '/Users/figo/pcharm/ml/try_sklearn/kevin_markham/l6/Advertising.csv',
        index_col=0)  # 第一列作为 dataframe 的 index列， shape 200 * 4
    return df


def create_model(feature_cols):  # start create model
    df = load_data()
    X_train, X_test, y_train, y_test = _process_data(df, feature_cols)

    linreg = LinearRegression().fit(X_train, y_train)
    return linreg, X_test, y_test


def _process_data(data, feature_cols):
    # X, 3 columns, without sales.

    X = data[feature_cols]  # <class 'pandas.core.frame.DataFrame'>, shape 200 * 3

    print('\nX只需要3列作为feature：')
    print(X.head())
    print(X.shape)

    print('\ny既response/target：')
    y = data['Sales']  # <class 'pandas.core.series.Series'>
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print(X_train.shape)

    return X_train, X_test, y_train, y_test


def _explain_model(model, feature_cols):
    """
    说明fit后的intercept 和 coefficient。
    """
    print('\n学习后得到的intercept 和 3个coefficient：')
    print(model.intercept_)
    print(model.coef_)

    # 0.0465645678742 * TV + ...
    coef_description = ' + '.join(['%s * %s' % coef_tup
                                   for coef_tup in zip(model.coef_, feature_cols)])

    print('\n既公式为:')
    # y = 2.88 + 0.0465645678742 * TV + 0.179158122451 * Radio + 0.00345046471118 * Newspaper
    print('y = %s + %s' % (model.intercept_, coef_description))

    print("""如何解释coefficient 0.0465 * TV ?:
        每在TV花一单元的费用(既$1000)，就可以提高sales 46.5 次
        """)


def _evaluation(y_test, y_pred):
    pass


def run():
    linreg1, X_test, y_test = create_model(feature_cols=['TV', 'Radio', 'Newspaper'])
    _explain_model(model=linreg1, feature_cols=['TV', 'Radio', 'Newspaper'])

    """
    RMAE1
    """
    y_pred = linreg1.predict(X_test)
    print('含Newspaper的错误率，较高: %s。' %
          np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)))

    """
    RMAE2
    """
    linreg2, X_test, y_test = create_model(feature_cols=['TV', 'Radio'])
    y_pred = linreg2.predict(X_test)
    print('不含Newspaper的错误率，较低: %s。' %
          np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)))

    print('结论，Newspaper导致了高错误率，不是一个好的feature，应该考虑从model中移除。'
          '(从fit后得出的Newspaper的coefficient也可以看出，Newspaper的关联度很低。)')


if __name__ == '__main__':
    run()
