# -*- encoding: utf-8 -*-
"""
@File    : transform.py
@Time    : 2020/4/12 15:13
@Author  : Alessia K
@Email   : ------
@Reference: https://github.com/facebookresearch/pycls/blob/master/pycls/
"""

from PIL import Image
import numpy as np
import math

def color_norm(img, mean, std):
    """Per-channel Normalization: [C, H, W]"""
    for i in range(img.shape[0]):
        img[i] = img[i] - mean[i]
        img[i] = img[i] / std[i]
    # print('color norm: ', img.shape)

    return img


def horizontal_flip(img, p, order = 'CHW'):
    """[C, H, W] or [H, W, C]"""
    assert order in ['CHW', 'HWC']
    if np.random.uniform() < p:
        if order == 'CHW':
            img = img[:, :, ::-1]
        else:
            img = img[:, ::-1, :]
    # print('horizontal: ', img.shape)
    return img


def zero_pad(img, pad_size):
    pad_width = ((0,0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(img, pad_width, mode='constant')


def random_crop(img, size, pad_size=0):
    """[C, H, W]"""
    if pad_size>0:
        img = zero_pad(img=img, pad_size=pad_size)
    h, w = img.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    img_crop = img[:, y : (y + size), x : (x+size)]
    # print('random_crop: ', img_crop.shape)
    return img_crop


def scale(img, size):
    """[H, W, C]"""
    h, w = img.shape[:2]
    if (w<=h and w == size) or (h<=w and h == size):
        return img
    h_new, w_new = size, size
    if w<h:
        h_new = int(math.floor((float(h) / w) * size))
    else:
        w_new = int(math.floor((float(w) / h) * size))
    img = Image.fromarray(np.uint8(img))
    img = img.resize((w_new, h_new), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    # print('scale: ', img.shape)
    return img


def center_crop(img, size):
    """[H, W, C]"""
    h, w = img.shape[:2]
    y = math.ceil((h - size)/2)
    x = math.ceil((w - size)/2)
    img_crop = img[y : (y + size), x : (x + size), :]
    # print('center crop: ', img_crop.shape)
    return img_crop


def random_sized_crop(img, size, area_frac=0.08, max_iter=10):
    """Inception-style cropping: [H, W, C]"""
    h, w = img.shape[:2]
    area = h * w
    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h_crop = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() <0.5:
            w_crop, h_crop = h_crop, w_crop
        if h_crop<=h and w_crop<=w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            img_crop = img[y : (y + h_crop), x : (x + w_crop), :]
            assert img_crop.shape[:2] == (h_crop, w_crop)
            img_crop = Image.fromarray(np.uint8(img_crop))
            img_crop = img_crop.resize((size, size), Image.BILINEAR)
            img_crop = np.array(img_crop, dtype=np.float32)
            # print('random size: ', img_crop.shape)
            return img_crop

    return center_crop(scale(img, size), size)


def lighting(img, alpha_std, eig_val, eig_vec):
    """AlexNet-style PCA jitter: [C, H, W]"""
    if alpha_std == 0:
        return img
    alpha = np.random.normal(0, alpha_std, size= (1,3))
    rgb = np.sum(eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0), axis=1)
    for i in range(img.shape[0]):
        img[i] = img[i] + rgb[2 - i]
    # print('lighting: ', img.shape)
    return img


