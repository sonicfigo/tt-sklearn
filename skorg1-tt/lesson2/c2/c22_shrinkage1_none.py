# coding=utf-8
"""
C22 shrink 几个文件，都是关于:
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

print("""X.shape(2, 1)
[ 0.5]
[ 1. ]""")
X = np.c_[.5, 1].T
assert (2, 1) == X.shape

y_list = [.5, 1]  # list

print("""test.shape(2,1)
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

print("""以下12个点，6条线的图形，说明几点：
1. 
2. 6条线，
""")
plt.figure()

regr = linear_model.LinearRegression()  # 没有任何 shrink 效果

"""
同一个model，fit 6次不同的 random 数据，predict 同一套 test 数据
图中 12小点，12大点，6线，说明：
1. 6线，是根据12大点，画出的，各自的形状不太一样，这个 model 不太稳定，会造成高 variance 现象
2. 6线，都穿过了12小点，学习的很好，这是一个低 bias 现象
"""
for _ in range(6):
    # 中位数0， standard deviation 1，模拟噪音
    # 有噪音，fit的数据稍有不同，画出了6条不同的线
    X_random = .1 * np.random.normal(size=(2, 1)) + X
    plt.scatter(X_random, y_list, s=3)  # 画随机fit的2个点

    regr.fit(X_random, y_list)  # fit 随机数据

    y_predict_each = regr.predict(X_test_fixed)
    plt.scatter(X_test_fixed, y_predict_each, s=31)  # 画固定test的2个点
    plt.plot(X_test_fixed, y_predict_each)  # 画predict的拟合线

plt.show()
