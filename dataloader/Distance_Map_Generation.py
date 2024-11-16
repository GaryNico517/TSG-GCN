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
import sympy as sp
import pandas as pd
import math
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from skimage.measure import regionprops,label
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """
    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis
                # print(np.max(posdis), np.min(posdis))
    return fg_dtm

def Distance_Map_Generation(mask_array):
    mask_new_array = np.zeros_like(mask_array).astype(np.float)
    for i in np.unique(mask_array):
        if i == 0:
            continue
        mask_copy_array = np.copy(mask_array)
        mask_copy_array[mask_copy_array != i] = 0
        mask_copy_array[mask_copy_array == i] = 1
        mask_copy_array = mask_copy_array[np.newaxis, :]
        dis_map = compute_dtm(mask_copy_array, mask_copy_array.shape)
        # dis_cls_map = dis_map[0] > dis_thresh
        mask_new_array += dis_map[0] / np.max(dis_map[0])
    # print(np.unique(mask_new_array))
    return mask_new_array