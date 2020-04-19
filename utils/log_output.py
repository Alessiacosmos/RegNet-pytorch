# -*- encoding: utf-8 -*-
"""
@File    : log_output.py
@Time    : 2020/4/15 19:43
@Author  : Alessia K
@Email   : ------
"""

import os
from datetime import datetime

def build_training_folder(basic_dir):
    foldername = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    folder_all = os.path.join(basic_dir, foldername)
    os.makedirs(folder_all, exist_ok=True)
    return folder_all

def get_log_filename(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.txt'
    return os.path.join(log_dir, filename)

def get_test_log_filename(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '_test.txt'
    return os.path.join(log_dir, filename)



