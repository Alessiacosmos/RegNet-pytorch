# -*- encoding: utf-8 -*-
"""
@File    : lr_policy.py
@Time    : 2020/4/15 15:14
@Author  : Alessia K
@Email   : ------
"""
import numpy as np

def lr_policy_steps(epoch_i, base_lr, milestone, lr_mult):
    """steps schedule"""
    ind = [i for i, s in enumerate(milestone) if epoch_i >= s][-1]
    lr = base_lr * (lr_mult ** ind)
    return lr

def lr_policy_exp(epoch_i, base_lr, gamma):
    """exponential schedule"""
    lr = base_lr * (gamma ** epoch_i)
    return lr

def lr_policy_cos(epoch_i, base_lr, max_epoch):
    """cosine schedule"""
    lr = 0.5 * base_lr * (1.0 + np.cos(np.pi * epoch_i / max_epoch))
    return lr
