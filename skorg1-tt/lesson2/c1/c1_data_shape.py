# coding=utf-8
"""
Statistical learning: the setting and the estimator object in scikit-learn

数据如何转化成sklearn可以处理的格式：2d矩阵(n_samples, n_features)
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

"""
iris
data 是可用的格式, 2d
"""
iris = datasets.load_iris()
data = iris.data
assert (150, 4) == data.shape
# print(iris.DESCR) #  说明文字

"""
digit
data 是sk可用的，2d的
images 是sk不可用的，3d的
"""
digits = datasets.load_digits()
assert (1797, 64) == digits.data.shape
assert (1797, 8, 8) == digits.images.shape

"""
image是 8 * 8的图像
"""
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
plt.show()

"""
可以转化image 为 64 vector，这样digits.images就可以被sklearn所用了。
具体咋用？没深入尝试。
"""
images_reshape = digits.images.reshape((digits.images.shape[0], -1))
assert (1797, 64) == images_reshape.shape  # 转为2d时，64 = 总数（1797 * 8 * 8）/ n_samples数

"""
estimator 最关键的接口:
    - fit(X, y)
    - y_ = predict(X)
"""
model1 = LinearRegression(fit_intercept=True)
# estimator.fit()
print(model1.fit_intercept)
print(model1.estimated_param_)
