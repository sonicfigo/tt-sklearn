# coding=utf-8
"""
Cross-validation to set a parameter can be done more efficiently on an algorithm-by-algorithm basis.

This is why, for certain estimators, scikit-learn exposes
Cross-validation:evaluating estimator performance estimators that
set their parameter automatically by cross-validation:

有些model具备 CV版本，这种版本的modle， 会通过 cross-validation，自动设置好参数。

"""

from sklearn import linear_model, datasets

lasso = linear_model.Lasso()
lasso_cv = linear_model.LassoCV()

diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

lasso.fit(X_diabetes, y_diabetes)
lasso_cv.fit(X_diabetes, y_diabetes)

# The estimator chose automatically its lambda:
print(lasso)

print(lasso_cv)

print(lasso_cv.alpha_)
