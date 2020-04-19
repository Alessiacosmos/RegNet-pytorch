# -*- encoding: utf-8 -*-
"""
@File    : optimizer.py
@Time    : 2020/4/15 15:21
@Author  : Alessia K
@Email   : ------
"""

import torch
import utils.lr_policy as lr_policy

def get_epoch_lr(epoch_i, cfg):
    lr_strategy   = cfg['OPTIM']['LR_POLICY']
    warmup_epochs = cfg['OPTIM']['WARMUP_EPOCHS']
    warmup_factor = cfg['OPTIM']['WARMUP_FACTOR']

    base_lr = cfg['OPTIM']['BASE_LR']
    if lr_strategy not in ['steps', 'exp', 'cos']:
        raise NotImplementedError('Undefined LR policy: ' + lr_strategy)
    lr = base_lr
    if lr_strategy == 'step':
        lr = lr_policy.lr_policy_steps(epoch_i, base_lr=base_lr,
                                       milestone=cfg['OPTIM']['STEPS'],
                                       lr_mult=cfg['OPTIM']['LR_MULT'])
    if lr_strategy == 'exp':
        lr = lr_policy.lr_policy_exp(epoch_i, base_lr=base_lr, gamma=cfg['OPTIM']['GAMMA'])
    if lr_strategy == 'cos':
        lr = lr_policy.lr_policy_cos(epoch_i, base_lr=base_lr, max_epoch=cfg['OPTIM']['MAX_EPOCH'])
    if epoch_i < warmup_epochs:
        alpha = epoch_i / warmup_epochs
        warmup_factor = warmup_factor * (1.0 - alpha) + alpha
        lr *= warmup_factor

    return lr

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


