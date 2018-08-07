# coding=utf-8
"""

"""

from PIL import Image
from tornado.log import app_log
from io import BytesIO
import PIL.ImageOps
import base64

logger = app_log
_THRESHOLD_TO_2D = 150


def simplify(file_like, _mode="gray"):
    ''' support 2 modes: gray / 2d '''
    logger.debug("converting file ...")
    img = Image.open(file_like)
    if img.mode == 'RGBA':  # for png file
        r, g, b, a = img.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        img = PIL.ImageOps.invert(rgb_img)
    if _mode == 'gray':
        fp = convert_gray(img)
    elif _mode == '2d':
        fp = convert_gray(img)
        fp = convert_2d(fp, _threshold=_THRESHOLD_TO_2D)
    else:
        raise ValueError("Invalid Mode")
    logger.debug("...simplify done")
    return fp


def convert_gray(img):
    fp = img.convert("L")
    return fp


def convert_2d(img, _threshold=80):
    threshold = _threshold
    table = [x >= threshold and 1 or 0 for x in range(256)]
    fp = img.point(table, '1')
    return fp


def split(file_like, _l_margin=22, _back_space=3, _front_space=5,
          _pieces=5, _width=22, _height=50, _horizontal=True, _mode="gray"):
    ''' Support msk_captcha only, it'size is 50 * 150, horizontal arranged !!
        Split image into '_pieces' with label (image should be labeled first)
    '''
    with Image.open(file_like) as img:
        if img.mode != '1' and img.mode != 'L':
            img = simplify(file_like, _mode=_mode)
        if _horizontal:
            blocks = [
                (_l_margin + x * _width - _back_space, 5,
                 _l_margin + (x + 1) * _width + _front_space, _height - 5)
                for x in range(_pieces)
                ]
        crop = lambda x: img.crop(x)
        regions = map(crop, blocks)
        logger.debug("...split done")
        return regions


def make_dataset(file_like):
    ''' make .csv file combine all img_file's data '''
    import numpy as np
    splited = split(file_like)
    data = [np.array(x).ravel() for x in splited]
    return np.stack(data)


def trans_from_b64str(stream):
    return BytesIO(base64.b64decode(stream))
