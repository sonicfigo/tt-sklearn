# coding=utf-8
"""
C22 shrink 几个例子，都是关于:
    目标：降低noise，提高test时的分数
    涉及：bias 与 variance 的高低权衡
既fit的学习数据，有noise，就会造成拟合出来的学习结果差别很大
目标，让不同的fit学习数据，拟合出来的学习结果，差别不大


LinearRegression，没有缩减特征，有噪音，6条拟合线分叉较大，体现了一个 低 bias，高variance 现象
说人话就是：
1. 不做shirink，6次学习，predict出来的成绩，都跟学的时候很符合，学的很好。
2. 但6次之间，长得不太一样，也就是同一套test 数据X_test(0 和 2)，y_predict值不尽相同。
3. 那么以后来了一个真实数据需要predict，误差会很大，因为指不定此时的model fit 了什么数据，到底拟合出来的线，是6条线的哪一条，每一条y_predict都差很多
也就是 fit 的数据稍微有变动，对最终拟合线影响很大

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

print("""
X.shape(2, 1)
[ 0.5]
[ 1. ]""")
X = np.c_[.5, 1].T
assert (2, 1) == X.shape

y_list = [.5, 1]  # list

print("""
test.shape(2,1)
[0]
[2]""")
X_test_fixed = np.c_[0, 2].T
assert (2, 1) == X_test_fixed.shape

"""
seed(0) 固定的随机值(既，6次之间是不同的，但每次执行，这6次与上6次是相同的)

就是让6次，都是
[ 0.5]
[ 1. ]
左右的不同的数据
"""
np.random.seed(0)

plt.figure(1, (13, 9))

regr = linear_model.LinearRegression()  # 没有任何 shrink 效果

"""
图中 12大点，12小点，6线，：
- 先通过大点学习，拟合出粗线
- 再考试，试试小点画出来的位置是在哪里
- 根据小点，拟合出细线
- 结论：
    1. 细线就是与粗线重合，证明考试，完全依赖之前的学习，低 bias 现象
    2. 6粗线各自之间长得很不一样，这个 model 不太稳定，高 variance 现象

实践意义：
1. 同一个model
2. fit 6次不同的 random 数据，出来6条很不一样的粗线
3. predict 同一套 test 数据，此时完全依赖于用的是6条粗线的哪一条。
既：fit稍微有点不同，test考试出来的结果就很不一样，model的泛化能力很差
"""


def _get_X_with_noise():
    """
    中位数0， standard deviation 1，模拟噪音，使得每次取得的 X_with_noise 都是在 0.5 和 1 左右晃荡
    """
    return .1 * np.random.normal(scale=1, size=(2, 1)) + X


for _ in range(6):
    """
    学习大点
    拟合粗线
    有噪音，fit的数据稍有不同，画出了6条不同的线
    """
    X_with_noise = _get_X_with_noise()
    plt.scatter(X_with_noise, y_list, s=31)  # 画fit的6 * 2个随机学习大点
    regr.fit(X_with_noise, y_list)  # fit 随机数据 vs y_list，这样两者就建立关系了
    plt.plot(X_with_noise, y_list, linewidth=6.0)

    """
    预测小点
    画出两点连线，可视化一下test时用的拟合线，对比一下粗线
    """
    y_predict_each = regr.predict(X_test_fixed)
    plt.scatter(X_test_fixed, y_predict_each, s=3)  # 画test的 6 * 2个固定测试小点
    plt.plot(X_test_fixed, y_predict_each, linewidth=1.0)  # 画predict的拟合线

    plt.ylim(-2, 4)

print('\n=================== X_test_fixed')
print(X_test_fixed)

plt.show()
