# coding=utf-8

"""
数据集2：数字

1797个数据 - digits.data
对应
1797个输出 - digits.target
对应
1797个图像 - digits.images
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

print('\ndigits.data - %s个输入数据, shape-%s。' % (len(digits.data), digits.data.shape))
print(digits.data)
print(type(digits.data))  # type:numpy.ndarray

print('\ndigits.target - %s 个target(文字答案)。' % len(digits.target))
print(digits.target)
print(type(digits.target))  # type:numpy.ndarray

print ('\n共%s张图片(图片答案)，每一张图片都是8*8的矩形，如图1矩阵：' % len(digits.images))
print(digits.images[0])

# 20% 或40%常见
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(digits.data,
                                                                         digits.target,
                                                                         digits.images,
                                                                         test_size=0.4)

print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))

print(len(img_train))
print(len(img_test))

print(X_train[0])
