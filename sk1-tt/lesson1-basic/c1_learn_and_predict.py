# coding=utf-8
"""
简单的一个SVC例子：
training 和 predict，并画出参考真实图像
"""
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)

# 共 (1797, 64), X_train 除去最后一个，为 (1796, 64)
X_train = digits.data[:-1]  # (1796, 64)
y_train = digits.target[:-1]  # (1796,)
clf.fit(X_train, y_train)

last_data = digits.data[-1:]
predict_value = clf.predict(last_data)
print('预测值是%s' % predict_value)

last_y = digits.target[-1:]
print('答案是%s，原图像如下:' % last_y)
last_img = digits.images[-1:][0]  # (8, 8)
plt.imshow(last_img, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
