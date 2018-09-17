# coding=utf-8
"""
SVM Exercise，分类 iris的其中两类(类1，类2)，feature 只取前两个：花萼length, 花萼width
并根据 decision_function ，而不是 predict 结果，来填充颜色

并不是所有数据都可以用线性隔开的，此时用上 kernel = 其他 试试
本例子演示了3种
    1. linear   线性核函数（linear kernel）            感觉分的太简单，只能直线分开
    2. poly     多项式核函数（ploynomial kernel）      感觉分的不好，且画得最慢
    3. rbf      (高斯)径向机核函数(radical basis function)  感觉分的最好
"""
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target

"""
用y的条件true，false
来取对应的X位置
"""
X = X[y != 0, :2]  # 只取 y!=0，既1，2两类对应的X， 前两个feature （前提是 X，y的 feature数 和 shape一样）
y = y[y != 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


def _plot_data_point():
    # 画学习的 fit 点
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

    # 画 test 点，圆圈里的是test数据
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)


def _plot_mesh_line():
    plt.axis('tight')  # 使得x，y轴的范围显示起来与数据相匹配

    # sepal length 的 min ~ max
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()

    # sepal width 的 min ~ max
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    """
    XX
    []
    """
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]  # j 是表示复数一部分，做啥用的？
    Z_decision_func = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # 画颜色，填充颜色, Put the result into a color plot
    Z_decision_func = Z_decision_func.reshape(XX.shape)  # (40000, ) -> (200, 200)

    # Z 是决策边界线，根据这个线，就能填充相应不同颜色
    plt.pcolormesh(XX, YY, Z_decision_func > 0, cmap=plt.cm.Paired)

    # 画出 "决策边界线 及 间距margin"
    plt.contour(XX, YY, Z_decision_func,
                # 以下list 3对应的位置：margin1， 边界线， margin2
                colors=['r', 'k', 'g'],  # 线颜色
                linestyles=['--', '-', '--'],  # 线风格
                levels=[-.5, 0, .5])  # 越大，线与点靠的越近


# fit the model
# for fig_num, kernel in enumerate(['linear', 'rbf', 'poly']):
for fig_num, kernel in enumerate(['rbf']):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()

    _plot_data_point()
    _plot_mesh_line()

    scores = clf.fit(X_train, y_train).score(X_test, y_test)
    plt.title('%s %s' % (kernel, scores))

plt.show()

"""
使用哪种内核的结论：链接：https://www.zhihu.com/question/21883548/answer/112128499

-----------------------------------
一般用线性核和高斯核，也就是Linear核与RBF核
需要注意的是需要对数据归一化处理，很多使用者忘了这个小细节
然后一般情况下RBF效果是不会差于Linear
但是时间上RBF会耗费更多，

其他同学也解释过了下面是吴恩达的见解：
    1. 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM
    2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM + Gaussian Kernel （本例子既是）
    3. 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

-----------------------------------
记口诀：

初级
    高维用线性，不行换特征；低维试线性，不行换高斯
中级
    线性试试看，不行换高斯，卡方有奇效，绝招MKL
玩家
    Kernel度量相似性，自己做啊自己做

"""
