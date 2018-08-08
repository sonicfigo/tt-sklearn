# coding=utf-8
"""
[learning_curve]

可以很直观的看出我们的 model 学习的进度,
对比发现有没有 overfitting 的问题.然后我们可以对我们的 model 进行调整,克服 overfitting 的问题.

什么是overfitting：是一个问题，过于纠结training数据准确度，拟合有三种状态:
    1. 不拟合 under fit     Θ0 + Θ1x                                不能很好拟合的一次函数直线
    2. 拟合 just right     Θ0 + Θ1x + Θ2x²                        不错的二次函数曲线
    3. 过度拟合 over fit    Θ0 + Θ1x + Θ2x² + Θ3x³ + Θ4x(四次方)   完全拟合全部的点的弯曲线

好的fitting，应该随着traning， 误差值一直减小

此节：描述overfitting出现的过程，用了什么参数，出现了overfitting的问题，此时就要克服overfitting
下节：描述如何解决overfitting
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve  # 查看整个学习的过程，减小loss的过程
from sklearn.datasets import load_digits
from sklearn.svm import SVC

digits = load_digits()
X = digits.data
y = digits.target

print('\n 共%s个samples' % len(X))

"""
learning_curve 参数详解(其实是用了 cross_val_score方式，见p1_3.ipynb):

1. model
    gamma=0.001   好
    gamma=0.01    不好-因为overfitting了
4. cv 分成十组
5. neg_mean_squared_error 方差值，对比的误差值
6. train_sizes 在 10%， 25%， 50%， 75%， 100% 五个点，记录误差值
"""
model = SVC(gamma=0.001)  # 好
# model = SVC(gamma=0.01)  # 不好，因为overfitting了, 能不能同样看看这个gamma 什么时候最好，可以，看下节怎么使用validation_curve

"""
learning curve 是什么原理？train比例多少，test比例多少？
根据官方doc文档：http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html

会拆分k次的train 和 test，（文档没有说清k是代表什么，但猜测应该就是 {train_sizes} 里的item个数。
每次train 都用来fit 模型，并用该模型重复测试 train，全新测试 test 数据，得到scoring
"""
train_sizes, \
train_loss, test_loss = learning_curve(model, X, y, cv=10,
                                       scoring='neg_mean_squared_error',  # 方差
                                       train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

print('\n train_sizes(5, )-%s，既那%s个记录点的百分比对应的学习个数。' % (train_sizes, len(train_sizes)))

print('\n===================误差结果')
print('\ntrain_loss shape-%s, \n%s。' % (train_loss.shape, train_loss))
print('\ntest_loss shape-%s, \n%s。' % (test_loss.shape, test_loss))

print('\n===================误差结果行的平均值。数据共5行，1行有10个数据(10个cv)，平均后变 (5, )，既1行只有1个数据')
train_loss_mean = -np.mean(train_loss, axis=1)  # 10组学习的误差平均值：没误差，因为学的，就是考的
test_loss_mean = -np.mean(test_loss, axis=1)  # 10组测试的误差平均值：应该要越来越低

print('学习误差 train_loss_mean shape-%s：\n%s' % (train_loss_mean.shape, train_loss_mean))
print('测试误差 test_loss_mean shape-%s: \n%s' % (test_loss_mean.shape, test_loss_mean))

# 红色：学习误差-几乎为0，因为学的本来就是对的。
plt.plot(train_sizes, train_loss_mean, 'o-', color="r", label="Training loss")

# 绿色：测试误差。
plt.plot(train_sizes, test_loss_mean, 'o-', color="g", label="Cross-validation loss")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
