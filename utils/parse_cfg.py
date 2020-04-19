# -*- encoding: utf-8 -*-
"""
@File    : parse_cfg.py
@Time    : 2020/4/13 15:07
@Author  : Alessia K
@Email   : ------
"""
import yaml

def load_cfg(cfg_path):
    with open(cfg_path) as f_cfg:
        cfg = yaml.load(f_cfg, Loader=yaml.FullLoader)

    assert cfg is not None, 'File Not Found ' + cfg_path

    return cfg
