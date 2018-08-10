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

X_train = digits.data[:-1]  # 必须用data，不能直接用digits.images
y_train = digits.target[:-1]
clf.fit(X_train, y_train)

predict_value = clf.predict(digits.data[-1:])
print('预测值是%s' % predict_value)

last_data = digits.data[-1:]
last_img = digits.images[-1:]
last_y = digits.target[-1:]
print('答案是%s，原图像如下:' % last_y)
plt.imshow(last_img[0], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
