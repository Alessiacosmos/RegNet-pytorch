# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 2020/4/15 22:50
@Author  : Alessia K
@Email   : ------
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.dataset import SingleDataset
from utils.parse_cfg import load_cfg
import utils.checkpoint as cp
import utils.log_output as log_g
import utils.evalutator as Evaluator

from model.AnyNet import AnyNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

def test(cfg):
    # basic settings
    loss_F = torch.nn.CrossEntropyLoss()

    gpu_nums = int(cfg['NUM_GPUS'])
    if gpu_nums == 0:
        use_cuda = False
    else:
        use_cuda = True

    # load model
    model = AnyNet(cfg)
    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

    # load_dataset
    Testpath = cfg['TEST']['PATH']
    RESIZE_SIZE_val = cfg['TEST']['IM_SIZE']
    test_data = SingleDataset(Testpath, split='val', resize_size=RESIZE_SIZE_val)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg['TEST']['BATCH_SIZE'],
                             shuffle=False, num_workers=cfg['DATA_LOADER']['NUM_WORKERS'], pin_memory=True)

    # optimizer and loss function and evaluator
    # optimizer and loss function and evaluator
    if cfg['OPTIM']['OPTIMIZER'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['OPTIM']['BASE_LR'], weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['OPTIM']['BASE_LR'], momentum=0.9, weight_decay=5e-4)

    # load checkpoint or initial weights
    if cfg['TEST']['WEIGHTS']:
        weights = cfg['TEST']['WEIGHTS']
        if not os.path.isfile(weights):
            raise RuntimeError("=> no checkpoint found at '{}'".format(weights))
        cp.load_checkpoint(cfg['TEST']['WEIGHTS'], gpu_nums, model)


    # save testing process
    log_file = log_g.get_test_log_filename(os.path.join(cfg['OUT_DIR'], 'log/'))
    log = open(log_file, 'w+')

    # start testing
    batch_size = cfg['TEST']['BATCH_SIZE']
    num_class = cfg['MODEL']['NUM_CLASSES']

    #############################################################################
    # start test
    #############################################################################
    model.eval()
    mean_loss_val = 0
    top1_sum = 0
    c_perclass = np.zeros(num_class)
    t_perclass = np.zeros(num_class)
    for val_epoch, (img_val, lbl_val) in enumerate(test_loader):
        if use_cuda:
            img_val, lbl_val = img_val.cuda(), lbl_val.cuda()

        # predict
        preds_val = model(img_val)

        # calculate loss
        loss_val = loss_F(preds_val, lbl_val)
        mean_loss_val += loss_val.item()

        # evaluation
        top1_acc, top2_acc = Evaluator.accuracy(preds_val, lbl_val, [1,2])
        # top1_err, top2_err = Evaluator.topk_error(preds_val, lbl_val, [1,2])
        correct, total = Evaluator.accuracy_perclass(preds_val, lbl_val, num_class)
        c_perclass += correct
        t_perclass += total
        top1_sum   += top1_acc

        print('[%s][%d/%d]--top1_acc: %.3f' % (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), val_epoch, len(test_loader), top1_acc))
    print('[{}]--mean_loss: {}\ttop1_acc: {}\tper_class_acc: {}'.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), mean_loss_val/len(test_loader), top1_sum/len(test_loader), float(c_perclass/t_perclass)*100))
    # save log
    log.write('[{}]--mean_loss: {}\ttop1_acc: {}\tper_class_acc: {}\n'.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),(mean_loss_val / len(test_loader)), top1_sum/len(test_loader), float(c_perclass/t_perclass)*100))

    log.close()


if __name__ == '__main__':
    # basic parmas
    cfg = load_cfg('data/AnyNet_cpu.yaml')
    test(cfg)







