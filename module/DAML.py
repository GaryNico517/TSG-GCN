import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import sys
import os
import matplotlib.pyplot as pl
from PIL import Image as Img
import random
import cv2

def one_hot_differentiable(inp, dim):
    # print('step1',inp,dim)
    y_soft = inp.softmax(dim=dim)
    # print('step2', y_soft)
    index = y_soft.max(dim, keepdim=True)[1]
    # print('step3', index)
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    # print('step4', y_hard)
    ret = y_hard - y_soft.detach() + y_soft
    # print('step5',ret)
    return ret

def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

def metric_calculate(x, y):
    tp, tn, fp, fn = 0, 0, 0, 0

    tp += torch.sum(x * y)
    tn += torch.sum((1 - x) * (1 - y))
    fp += torch.sum((1 - x) * y)
    fn += torch.sum(x * (1 - y))

    dice_tptn = (2 * tp ) / (2 * tp + fp + 1 * fn + 10e-4)
    return float(tp)

def AM_Generate_GT(mip_mask):
    MIP_mask_temp = torch.tensor(mip_mask[:],requires_grad=False)
    MIP_mask = soft_dilate(MIP_mask_temp.unsqueeze(1)).squeeze()
    background = torch.clone(MIP_mask[0]).detach()
    with torch.no_grad():
        MIP_mask[0] = MIP_mask[0] - background

    # staff = torch.clone(MIP_mask[-1]).detach()
    # with torch.no_grad():
    #     MIP_mask[-1] = MIP_mask[-1] - staff

    MIP_1 = MIP_mask.view(MIP_mask.shape[0],-1)
    MIP_2 = MIP_1.transpose(0,1)
    gram_matrix = torch.mm(MIP_1, MIP_2)
    weight = torch.clone(gram_matrix).detach()
    diag = torch.diag(torch.diag(weight))
    tp = gram_matrix / torch.max(weight - diag)
    tp[tp > 0] = 1
    return tp


def AM_Generate_Pred(mip_mask_input):
    if type(mip_mask_input) == np.ndarray:
        mip_mask = torch.tensor(mip_mask_input[:], requires_grad=False)
    else:
        mip_mask = torch.clone(mip_mask_input)
    MIP_mask = soft_dilate(mip_mask)

    background = torch.clone(MIP_mask[:,0]).detach()
    with torch.no_grad():
        MIP_mask[:,0] = MIP_mask[:,0] - background

    # staff = torch.clone(MIP_mask[:,-1]).detach()
    # with torch.no_grad():
    #     MIP_mask[:,-1] = MIP_mask[:,-1] - staff
    # print(MIP_mask.shape)
    MIP_1 = MIP_mask.view(MIP_mask.shape[0],MIP_mask.shape[1],-1)
    MIP_2 = MIP_1.transpose(1,2)
    # print(MIP_1.shape,MIP_2.shape)
    gram_matrix = torch.bmm(MIP_1, MIP_2)

    weight = torch.clone(gram_matrix).detach()
    diag = torch.zeros_like(weight)
    for diag_idx in range(weight.shape[0]):
        diag[diag_idx] = torch.diag(torch.diag(weight[diag_idx]))

    tp = gram_matrix / torch.max(weight - diag)

    # print('tp',tp)
    thresh_channel = torch.clone(tp).detach()
    thresh_channel[thresh_channel > -1] = 0.05
    tp_thresh = torch.cat((tp.unsqueeze(1), thresh_channel.unsqueeze(1)), dim=1)
    # print('thresh',tp_thresh)
    tp_thresh = one_hot_differentiable(tp_thresh, dim=1)

    return tp_thresh[:,0]

def AM_Generate_Pred_nograd(mip_mask_input):
    if type(mip_mask_input) == np.ndarray:
        mip_mask = torch.tensor(mip_mask_input[:], requires_grad=False)
    else:
        mip_mask = torch.clone(mip_mask_input)
    MIP_mask = soft_dilate(mip_mask)

    background = torch.clone(MIP_mask[:,0]).detach()
    with torch.no_grad():
        MIP_mask[:,0] = MIP_mask[:,0] - background

    # staff = torch.clone(MIP_mask[:,-1]).detach()
    # with torch.no_grad():
    #     MIP_mask[:,-1] = MIP_mask[:,-1] - staff
    # print(MIP_mask.shape)
    MIP_1 = MIP_mask.view(MIP_mask.shape[0],MIP_mask.shape[1],-1)
    MIP_2 = MIP_1.transpose(1,2)
    # print(MIP_1.shape,MIP_2.shape)
    gram_matrix = torch.bmm(MIP_1, MIP_2)

    weight = torch.clone(gram_matrix).detach()
    diag = torch.zeros_like(weight)
    for diag_idx in range(weight.shape[0]):
        diag[diag_idx] = torch.diag(torch.diag(weight[diag_idx]))

    tp = gram_matrix / torch.max(weight - diag)

    # print('tp',tp)
    thresh_channel = torch.clone(tp).detach()
    thresh_channel[thresh_channel > -1] = 0.05
    tp_thresh = torch.cat((tp.unsqueeze(1), thresh_channel.unsqueeze(1)), dim=1)
    # print('thresh',tp_thresh)
    tp_thresh = one_hot_differentiable(tp_thresh, dim=1)

    return tp_thresh[:,0].detach()

def ThreeD_Seg_to_TwoD_Proj(mask):
    mask_oh = torch.clone(mask)
    background = torch.clone(mask_oh[:][0]).detach()
    with torch.no_grad():
        mask_oh[:][0] = 1 - background

    mip_max_mask = torch.max(mask_oh, dim=2).values

    background_mip = torch.clone(mip_max_mask[:][0]).detach()
    with torch.no_grad():
        mip_max_mask[:][0] = 1 - background_mip
    return mip_max_mask