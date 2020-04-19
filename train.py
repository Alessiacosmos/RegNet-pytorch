# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2020/4/13 17:10
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
import utils.optimizer as optim
import utils.checkpoint as cp
import utils.log_output as log_g
import utils.evalutator as Evaluator

from model.AnyNet import AnyNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initial weights
def init_weights(model, zero_init_gamma=False):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.001)


def main(cfg):
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
    Trainpath   = cfg['TRAIN']['PATH']
    RESIZE_SIZE = cfg['TRAIN']['IM_SIZE']
    train_data  = SingleDataset(Trainpath, split='train', resize_size=RESIZE_SIZE)
    train_loader= DataLoader(dataset=train_data, batch_size=cfg['TRAIN']['BATCH_SIZE'],
                             shuffle=True, num_workers=cfg['DATA_LOADER']['NUM_WORKERS'], pin_memory=True)

    Testpath    = cfg['TEST']['PATH']
    RESIZE_SIZE_val = cfg['TEST']['IM_SIZE']
    test_data   = SingleDataset(Testpath, split='val', resize_size=RESIZE_SIZE_val)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg['TEST']['BATCH_SIZE'],
                             shuffle=False, num_workers=cfg['DATA_LOADER']['NUM_WORKERS'], pin_memory=True)

    # optimizer and loss function and evaluator
    if cfg['OPTIM']['OPTIMIZER'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['OPTIM']['BASE_LR'], weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['OPTIM']['BASE_LR'], momentum=0.9, weight_decay=5e-4)

    # load checkpoint or initial weights
    start_epoch = 0
    if cfg['TRAIN']['RESUME'] is not None:
        resume = cfg['TRAIN']['RESUME']
        if not os.path.isfile(resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(resume))
        checkpoint_epoch = cp.load_checkpoint(resume, gpu_num=gpu_nums, model=model, optimizer=optimizer)
        start_epoch = checkpoint_epoch + 1
    elif cfg['TRAIN']['WEIGHTS']:
        cp.load_checkpoint(cfg['TRAIN']['WEIGHTS'], gpu_nums, model)
    else:
        init_weights(model, zero_init_gamma=cfg['BN']['ZERO_INIT_FINAL_GAMMA'])

    # save training process
    log_file = log_g.get_log_filename(os.path.join(cfg['OUT_DIR'], 'log/'))
    log = open(log_file, 'w+')

    # start training
    max_epoch   = cfg['OPTIM']['MAX_EPOCH']
    batch_size  = cfg['TRAIN']['BATCH_SIZE']
    eval_period = cfg['TRAIN']['EVAL_PERIOD']
    batch_count = 0
    total_step  = len(train_loader)
    num_class   = cfg['MODEL']['NUM_CLASSES']
    # correct_all = list(0. for i in range(cfg['MODEL']['NUM_CLASSES']))
    # total_all   = list(0. for i in range(cfg['MODEL']['NUM_CLASSES']))
    for epoch in range(start_epoch, max_epoch):
        print('**************train --%d-- **************' % (epoch))
        log.write('**************train --%d-- **************\n' % (epoch))

        # update learning rate
        lr = optim.get_epoch_lr(epoch_i=epoch, cfg=cfg)
        optim.set_lr(optimizer, lr)

        #############################################################################
        # start training an epoch
        #############################################################################
        model.train()
        c_train = 0
        t_train = 0
        for i, (img, lbl) in enumerate(train_loader):
            batch_count += 1

            # use cuda
            if use_cuda:
                img, lbl = img.cuda(), lbl.cuda()

            # forward
            preds = model(img)
            loss = loss_F(preds, lbl)

            # backward
            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if (batch_count % batch_size) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            _, predicted = preds.max(1)
            c_train += predicted.eq(lbl).sum().item()
            t_train += lbl.size(0)

            # print epoch, step, loss, lr
            print('[%s]--train: %d/%d\tstep:%d/%d----lr:%.5f---loss:%.4f---Acc:%.3f' % (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                (epoch + 1), max_epoch, (i + 1), total_step, lr, loss.item(), 100*(c_train/t_train)))
            log.write('[%s]--train: [%d/%d]\tstep: [%d/%d]\t----lr:%.5f---loss:%.4f---Acc:%.3f\n' %(
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      (epoch + 1), max_epoch, (i + 1), total_step, lr, loss.item(), 100*(c_train/t_train)))


        #############################################################################
        # start validation
        #############################################################################
        if ((epoch+1) % eval_period == 0):
            print('**************validation --%d-- **************' % ((epoch + 1) // eval_period))
            model.eval()
            mean_loss_val = 0
            correct = np.zeros((num_class))
            total = np.zeros((num_class))
            top1_acc_sum = []
            with torch.no_grad():
                for val_epoch, (img_val, lbl_val) in enumerate(test_loader):
                    if use_cuda:
                        img_val, lbl_val = img_val.cuda(), lbl_val.cuda()

                    # predict
                    preds_val = model(img_val)

                    # calculate loss
                    loss_val  = loss_F(preds_val, lbl_val)
                    mean_loss_val += loss_val.item()

                    # evaluation
                    top1_acc, top2_acc = Evaluator.accuracy(preds_val, lbl_val, [1,2])
                    correct_i, total_i = Evaluator.accuracy_perclass(preds_val, lbl_val, num_class)
                    correct += correct_i
                    total   += total_i
                    top1_acc_sum.append(top1_acc)

                    print('[%s]--valid: [%d/%d]\tloss: %.4f---top1_acc: %.3f' % (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        val_epoch, len(test_loader), loss_val.item(), top1_acc.item()))
                print('[{}]--valid: [{}]\tmean_loss: {}\ttop1_acc: {}\tper_class_acc: {}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    (epoch + 1), (mean_loss_val / len(test_loader)), np.mean(top1_acc_sum), 100*(correct/total)))
                # save log
                log.write('[{}]--valid: [{}]\tmean_loss: {}\ttop1_acc: {}\tper_class_acc: {}\n'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    (epoch + 1), (mean_loss_val / val_epoch), np.mean(top1_acc_sum), 100*(correct/total)))


        #############################################################################
        # save model
        #############################################################################
        if ((epoch+1)%5==0):
            checkpoint_file = os.path.join(cfg['OUT_DIR'], 'checkpoint/')
            checkpoint_filename = cp.save_checkpoint(model, optimizer, epoch, gpu_nums, checkpoint_file)
            log.write('[{}]--save checkpoint: {}\n'.format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                checkpoint_filename
            ))

    log.close()



if __name__ == '__main__':
    # basic parmas
    cfg = load_cfg('data/AnyNet_cpu.yaml')
    main(cfg)






