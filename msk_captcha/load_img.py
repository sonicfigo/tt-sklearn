# coding=utf-8
"""
wlh提供的马士基验证码接口
"""
from io import BytesIO

from msk_captcha.img_helper import make_dataset

imgpath = '/Users/figo/pcharm/ml/try_sklearn/msk_captcha/ZhGuT.png'

# wlh原始调用方法，图片的base64字符串
# def read_img_base64():
#     with open(imgpath, 'rb') as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#         return encoded_string
#
#
# base64_str = read_img_base64()
# file_like = base64.b64decode(base64_str)
#
# imgdata = BytesIO(file_like)
# dataset = make_dataset(imgdata)
# print(dataset)
# assert (5, 1200) == dataset.shape  # 5行 1200列

# 简易写法
with open(imgpath, 'rb') as image_file:
    imgdata = BytesIO(image_file.read())
    dataset = make_dataset(file_like=imgdata)
    print(dataset)
    print(type(dataset))
    assert (5, 1200) == dataset.shape  # 5行 1200列
