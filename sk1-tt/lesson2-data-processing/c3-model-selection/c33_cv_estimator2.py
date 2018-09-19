# coding=utf-8
"""

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)  # alphas 变动的范围，10的4次方 ~ 10的负0.5次方，30个点
tuned_parameters = [{'alpha': alphas}]
n_folds = 3

gscv_lasso = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
gscv_lasso.fit(X, y)

scores = gscv_lasso.cv_results_['mean_test_score']  # 平均分数
scores_std = gscv_lasso.cv_results_['std_test_score']  # 标准差


def plt_sth():
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)  # 图形画出来的x表示， alphas的对数刻度

    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(n_folds)

    plt.semilogx(alphas, scores + std_error, 'b--')
    plt.semilogx(alphas, scores - std_error, 'b--')

    # 填充两块淡蓝色
    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

    plt.xlabel('each grid = label value (alpha)\'s natural logarithm')
    plt.ylabel('CV score +/- std error')

    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([alphas[0], alphas[-1]])

plt_sth()

"""
# #############################################################################
Bonus: how much can you trust the selection of alpha?

To answer this question we use the LassoCV object that sets its alpha
parameter automatically from the data by internal cross-validation 
(i.e. it performs cross-validation on the training data it receives).

We use external cross-validation to see how much the automatically obtained
alphas differ across different cross-validation folds.

"""
lasso_cv = LassoCV(alphas=alphas, random_state=0)
k_fold = KFold(3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")

"""
答案:
无法得到可信的alpha，原因：
    1. LassoCV fit不同的train数据，自动得到的 alpha 都不尽相同
    2. 使用这些 alpha，考试得到的分数也差异很大
"""
plt.show()
