# -*- encoding: utf-8 -*-
"""
@File    : evalutator.py
@Time    : 2020/4/15 20:26
@Author  : Alessia K
@Email   : ------
@Reference: https://github.com/facebookresearch/pycls/blob/master/pycls/
"""
import torch
import numpy as np

def topk_correct(pred, target, topk):
    """Computes the accuracy of the correct top-k predictions for the specified values of k"""

    assert pred.size(0) == target.size(0), "Batch dim of predictions and targets must match"
    with torch.no_grad():
        max_k = max(topk)

        top_max_k_vals, top_max_k_inds = torch.topk(pred, max_k, dim=1, largest=True, sorted=True)

        # (batch_size, max_k) -> (max_k, batch_size)
        top_max_k_inds = top_max_k_inds.t()
        # (batch_size, ) -> (max_k, batch_size)
        rep_max_k_lbls = target.view(1, -1).expand_as(top_max_k_inds)

        top_max_k_correct = top_max_k_inds.eq(rep_max_k_lbls)

        # compute the number of top-k correct predictions
        topk_correct_list = []
        for k in topk:
            correct_k = top_max_k_correct[:k, :].view(-1).float().sum()
            topk_correct_list.append(correct_k)

        return topk_correct_list

def topk_error(pred, target, topk):
    """top-k error"""
    correct_k = topk_correct(pred, target, topk)
    batchsize = pred.size(0)
    topk_error_list = [(1.0 - x / batchsize) * 100 for x in correct_k]

    return topk_error_list

def accuracy(pred, target, topk):
    """accuracy = top-k correct"""
    correct_k = topk_correct(pred, target, topk)
    accuracy_list = [(x / pred.size(0)) * 100.0 for x in correct_k]

    return accuracy_list

def accuracy_perclass(pred, target, num_class):
    pred_top1 = torch.argmax(pred, 1)
    res = (pred_top1 == target).squeeze()

    correct = np.zeros((num_class))
    total   = np.zeros((num_class))

    for i in range(target.size(0)):
        cls = target[i]
        correct[cls] += res[i].item()
        total[cls] += 1

    return correct, total

# def accuracy_basic(pred, target):
#     predict = torch.argmax(pred, 1)
#     correct = (predict == target).sum().item()
#     total   = target.size(0)









