# -*- encoding: utf-8 -*-
"""
@File    : checkpoint.py
@Time    : 2020/4/15 17:17
@Author  : Alessia K
@Email   : ------
"""
import os
import torch

def save_checkpoint(model, optimizer, epoch, gpu_num, checkpoint_file):
    os.makedirs(checkpoint_file, exist_ok=True)
    filename = 'train_checkpoint_%d.pt' % (epoch)
    checkpoint_filename = os.path.join(checkpoint_file, filename)
    state_dict = model.module.state_dict() if gpu_num > 1 else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_filename)
    return checkpoint_filename

def load_checkpoint(checkpoint_file, gpu_num, model, optimizer=None):
    assert os.path.exists(checkpoint_file), "Checkpoint '{}' not found".format(checkpoint_file)
    # load checkpoint on CPU
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # if multi-gpus
    model_state = model.module() if gpu_num > 1 else model
    model_state.load_state_dict(checkpoint['state_dict'])
    # load optimizer
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


