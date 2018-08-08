# coding=utf-8
"""

"""
from sklearn import datasets

import matplotlib.pyplot as plt

# 載入數字資料集
digits = datasets.load_digits()

# 畫出第一個圖片
plt.figure(1, figsize=(3, 3))
idx = 123
nda_img_x = digits.images[idx]
print(type(nda_img_x))
plt.imshow(nda_img_x, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
print(digits.target[idx])  # 看图像和此答案相符不
