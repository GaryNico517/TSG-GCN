import os
import random
import time
import traceback
import warnings
import copy
import torch
from skimage import measure, color
import SimpleITK as sitk
from torch.nn import functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
import math
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from skimage.measure import regionprops,label
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from torch.optim.lr_scheduler import _LRScheduler

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

class CosineAnnealingLRWithRestarts(_LRScheduler):
    """
    Cosine annealing with restarts.
    """
    def __init__(self, optimizer, T_0, T_mult=1.0, eta_min=0.0, k=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.k = k
        self.current_epoch = 0
        self.restarted = False
        self.cycle_len = T_0
        self.cycle_num = 0
        super(CosineAnnealingLRWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.restarted:
            self.current_epoch = 0
            self.restarted = False
            self.cycle_len = int(self.cycle_len * self.T_mult)
            self.cycle_num += 1
            self.base_lrs = [self.k * base_lr for base_lr in self.base_lrs]

        cos_inner = math.pi * (self.current_epoch % self.cycle_len)
        cos_inner /= self.cycle_len
        new_lr = self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (1 + math.cos(cos_inner))

        self.current_epoch += 1
        if self.current_epoch == self.cycle_len:
            self.restarted = True

        return [new_lr for _ in self.base_lrs]


