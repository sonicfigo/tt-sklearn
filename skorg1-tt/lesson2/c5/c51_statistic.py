# coding=utf-8
"""

"""

from sklearn.datasets import fetch_lfw_people

lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
"""
原始的图片                250 * 250
(default) slice 截取感兴趣部分：上下(70 ~ 195), 左右(78 ~ 172)
截图后图片                125 * 94
(default) resize 0.4     50 * 37 
"""
print(lfw.images.shape)
print(lfw.data.shape)
