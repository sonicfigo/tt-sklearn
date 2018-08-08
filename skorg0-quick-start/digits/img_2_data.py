# coding=utf-8
"""

"""

import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
images = digits.images


def run():
    plt.subplot(2, 4, 1 + 1)
    plt.axis('off')
    plt.imshow(images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %s' % 'title123')
    plt.show()


run()  # go
