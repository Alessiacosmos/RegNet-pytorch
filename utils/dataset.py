# -*- encoding: utf-8 -*-
"""
@File    : dataset.py
@Time    : 2020/4/1 0:29
@Author  : Alessia K
@Email   : ------
"""
import numpy as np
import os
from PIL import Image

import data.dict_class as cls_dict
import utils.transform as transform

import torch
from torch.utils import data as Data

# for normalization
_img_mean = (0.485, 0.456, 0.406)
_img_std = (0.229, 0.224, 0.225)


# for lighting augment
# reference: https://github.com/facebookresearch/pycls/blob/master/pycls/
_eig_vals = np.array([[0.2175, 0.0188, 0.0045]])
_eig_vecs = np.array([[-0.5675, 0.7192, 0.4009],
                      [-0.5808, -0.0045, -0.8140],
                      [-0.5836, -0.6948, 0.4203]])

def get_mean_and_std(dataset):
    dataloader = Data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    print('computing mean and std')
    for img, lbl in dataloader:
        for i in range(3):
            mean[i] += img[:, i, :, :].mean()
            std[i]  += img[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def load_image(filepath):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    image = Image.open(filepath)

    # If grayscale. Convert to RGB for consistency.
    if (image.mode != 'RGB'):
        image = image.convert('RGB')

    image = np.array(image)

    return image


# ****************************************************************************
# SingleDatasets
# load images for training model
# ****************************************************************************
class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, path, split, resize_size):
        with open(path, 'r') as file:
            self.img_files = file.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        assert len(self.img_files) > 0, 'No images found in %s' % path

        self.resize_size = resize_size
        self.split = split

        print('Number of images: ', len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index].split('\t')[0]

        # load images
        img = load_image(img_path)
        assert img is not None, 'File Not Found ' + img_path

        # augment #################################################################################

        if self.split == 'train':
            img = transform.random_sized_crop(img=img, size=self.resize_size, area_frac=0.08)
            img = transform.horizontal_flip(img=img, p=0.5, order='HWC')
        else:
            img = transform.scale(img=img, size=self.resize_size)
            img = transform.center_crop(img=img, size=self.resize_size)

        # HWC -> CHW
        img = img.transpose([2, 0, 1])

        # normalization
        img = img / 255.0
        img = transform.color_norm(img=img, mean=_img_mean, std=_img_std)
        # PCA jitter
        if self.split == 'train':
            img = transform.lighting(img=img, alpha_std=0.1, eig_val=_eig_vals, eig_vec=_eig_vecs)
        # img = transform.color_norm(img=img, mean=_img_mean, std=_img_std)

        # get labels for corresponding images
        # label_name = os.path.split(os.path.split(img_path)[0])[-1]
        # label = cls_dict.class_name[label_name]
        label = int(self.img_files[index].split('\t')[-1])
        return torch.from_numpy(img).float(), label







