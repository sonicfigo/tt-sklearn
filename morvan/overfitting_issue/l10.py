# coding=utf-8
"""
[validation_curve]
上一节尝试手工修改了 gamma=0.01 ，发现有overfitting，那么如何全面了解，多少 gamma 才是最好呢？

连续三节的cross validation让我们知道在机器学习中 validation 是有多么的重要,

如何解决overfitting？
发现overfitting分界点的参数，如gamma在0.0005。

[validation_curve]

这一次的 sklearn 中我们用到了 sklearn.learning_curve 当中的另外一种, 叫做 validation_curve。
用这一种 curve 我们就能更加直观看出改变 model 中的参数的时候有没有 overfitting 的问题了。
这也是可以让我们更好的选择参数的方法。
"""

from __future__ import print_function
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

# logspac用于创建等比数列
param_range = np.logspace(-6, -2.3, 5)  # log(-6) ~ log(-2.3)范围内等比取5个点，作为横坐标x
print ('param_range is %s ' % param_range)

# 定义了改变SVC()的gamma值， 范围为param_range
train_loss, test_loss = validation_curve(SVC(), X, y, param_name='gamma',
                                         param_range=param_range, cv=10,
                                         scoring='neg_mean_squared_error')

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()


# 结果图形解释：
# gamma超过0.001左右以后，training越来越好， test越来越差，就是明显的overfit。
#
# 整个过程的目的：
# 使用validation_curve，选取model的某个参数(如gamma)哪个值是最好的，而且不会出现overfitting的
# 情况。比如最后，我们就确定gamma大概是0.0006是最好的
