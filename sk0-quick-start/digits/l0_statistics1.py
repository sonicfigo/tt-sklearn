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

# (1797, 64)
print('\ndigits.data - %s个输入数据, shape-%s。' % (len(digits.data), digits.data.shape))
print(digits.data)
print(type(digits.data))  # type:numpy.ndarray

print('\ndigits.data[0] & images[0]，data 就是 image 的 8*8 数据打平成 64')
print(digits.data[0])
print(digits.images[0])

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

print(len(X_train))  # 1078 = 1797 * 0.6
print(len(X_test))  # 719 = 1797 * 0.4

print(len(y_train))
print(len(y_test))

print(len(img_train))
print(len(img_test))

print('\n===================X 的一条数据例子，注意shape 是 (64, )，而不是图片的shape(8, 8)')
print(X_train[0])
print(digits.data[1].ravel())
