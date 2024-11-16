# -*- coding: utf-8 -*-

import torch

from torch import nn
from torch.nn.functional import max_pool3d
import numpy as np
from torch.nn import functional as F

class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class Huber_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.thresh = 0.5
    def forward(self, y_true, y_pred):
        T = torch.abs(y_true - y_pred)
        Huber = torch.Tensor(T)
        Huber[T <= self.thresh] = 0.5 * torch.pow(T[T <= self.thresh],2)
        Huber[T > self.thresh] = self.thresh * (T[T > self.thresh] - self.thresh * 0.5)
        return torch.mean(Huber)