import os
import random
import time
import traceback
import warnings
import copy
import torch
from skimage import measure, color
import SimpleITK as sitk
import numpy as np
import sympy as sp
import pandas as pd
import math
from skimage import io, transform

from skimage.measure import regionprops,label
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian3D_isotropic((diameter ,diameter ,diameter), sigma=diameter/3)

    x, y, z = int(center[2]), int(center[1]), int(center[0])

    depth, height, width = heatmap.shape[0:3]

    left, right = min(x, radius), min(width - x, radius + 1)
    front, back = min(y, radius), min(height - y, radius + 1)
    top, bottom = min(z, radius), min(depth - z, radius + 1)
    masked_heatmap  = heatmap[z - top:z + bottom, y - front:y + back, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - front:radius + back, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap



def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m: m +1 ,-n: n +1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian3D_isotropic(shape, sigma=1):
    d, h, w = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-d: d +1 ,-h: h +1 ,-w: w +1]

    h = np.exp(-(x * x + y * y+ z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius_3d_anisotropic(det_size, min_overlap=0.1):
    d, h, w = det_size

    # r1
    r = sp.Symbol('r', real=True)
    a1 = 8
    b1 = -4 * (d + h + w)
    c1 = 2 * (w * h + w * d + h * d)
    d1 = (min_overlap - 1) * w * h * d
    f = a1 * r ** 3 + b1 * r ** 2 + c1 * r + d1
    r = sp.solve(f)
    r = np.asarray(r)
    index = np.where((r > 0) & (r < min(d / 2, h / 2, w / 2)))
    try:
        r1 = float(r[index])
    except Exception as e:
        # print(e)
        # print("r1",d, h, w, r)
        r1 = min((d - 1) / 2, (h - 1) / 2, (w - 1) / 2)

    # r2
    r = sp.Symbol('r', real=True)
    a2 = 8 * min_overlap
    b2 = 4 * (d + h + w) * min_overlap
    c2 = 2 * (w * h + w * d + h * d) * min_overlap
    d2 = (min_overlap - 1) * w * h * d
    f = a2 * r ** 3 + b2 * r ** 2 + c2 * r + d2
    r = sp.solve(f)
    r = np.asarray(r)
    index = np.where(r > 0)
    try:
        r2 = float(r[index])
    except Exception as e:
        # print(e)
        # print("r2",d, h, w, r)
        r2 = min((d - 1) / 2, (h - 1) / 2, (w - 1) / 2)

    # r3
    r = sp.Symbol('r', real=True)
    a3 = 1 + min_overlap
    b3 = -1 * (1 + min_overlap) * (d + h + w)
    c3 = (1 + min_overlap) * (w * h + w * d + h * d)
    d3 = (min_overlap - 1) * w * h * d
    f = a3 * r ** 3 + b3 * r ** 2 + c3 * r + d3
    r = sp.solve(f)
    r = np.asarray(r)
    index = np.where((r > 0) & (r < min(d, h, w)))
    try:
        r3 = float(r[index])
    except Exception as e:
        # print(e)
        # print("r3",d, h, w, r)
        r3 = min((d - 1) / 2, (h - 1) / 2, (w - 1) / 2)

    return min(r1, r2, r3)


def gaussian_radius_3d(det_size, min_overlap=0.1):
    d, h, w = det_size

    # r1
    r = sp.Symbol('r', real=True)
    a1 = 8
    b1 = -4 * (d + h + w)
    c1 = 2 * (w * h + w * d + h * d)
    d1 = (min_overlap - 1) * w * h * d
    f = a1 * r ** 3 + b1 * r ** 2 + c1 * r + d1
    r = sp.solve(f)
    r = np.asarray(r)
    index = np.where((r > 0) & (r < min(d / 2, h / 2, w / 2)))
    try:
        r1 = float(r[index])
    except Exception as e:
        # print(e)
        # print("r1",d, h, w, r)
        r1 = min((d - 1) / 2, (h - 1) / 2, (w - 1) / 2)

    # r2
    r = sp.Symbol('r', real=True)
    a2 = 8 * min_overlap
    b2 = 4 * (d + h + w) * min_overlap
    c2 = 2 * (w * h + w * d + h * d) * min_overlap
    d2 = (min_overlap - 1) * w * h * d
    f = a2 * r ** 3 + b2 * r ** 2 + c2 * r + d2
    r = sp.solve(f)
    r = np.asarray(r)
    index = np.where(r > 0)
    try:
        r2 = float(r[index])
    except Exception as e:
        # print(e)
        # print("r2",d, h, w, r)
        r2 = min((d - 1) / 2, (h - 1) / 2, (w - 1) / 2)

    # r3
    r = sp.Symbol('r', real=True)
    a3 = 1 + min_overlap
    b3 = -1 * (1 + min_overlap) * (d + h + w)
    c3 = (1 + min_overlap) * (w * h + w * d + h * d)
    d3 = (min_overlap - 1) * w * h * d
    f = a3 * r ** 3 + b3 * r ** 2 + c3 * r + d3
    r = sp.solve(f)
    r = np.asarray(r)
    index = np.where((r > 0) & (r < min(d, h, w)))
    try:
        r3 = float(r[index])
    except Exception as e:
        # print(e)
        # print("r3",d, h, w, r)
        r3 = min((d - 1) / 2, (h - 1) / 2, (w - 1) / 2)

    return max(r1, r2, r3)

def label_bbox_obtain(label_array,i):
    mask_voxel_coords = np.where(label_array == i)

    # print(mask_voxel_coords)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [maxzidx-minzidx, maxxidx-minxidx, maxyidx-minyidx]
    return bbox


def draw_gaussian_from_mask(case,mask):
        hm_size = mask.shape
        gaussian_hm  = np.zeros(hm_size, dtype=np.float32)
        heatmap = np.zeros(hm_size, dtype=np.float32)
        for i in np.unique(mask):
            if i == 0:
                continue
            label_i = np.copy(mask)
            bbox = label_bbox_obtain(label_i, i)
            radius = gaussian_radius_3d(bbox, min_overlap=0.1)  # 不变
            radius = max(0, int(radius))

            label_i[label_i != i] = 0
            region = measure.regionprops(label_i.astype(np.uint8))
            center_point = region[0].centroid
            draw_umich_gaussian(gaussian_hm, center_point, radius)
            gaussian_hm[label_i == 0] = 0.0  ##高斯图逻辑可以修改
            heatmap += gaussian_hm
        gaussian_hm = heatmap
        # new_mask = sitk.GetImageFromArray(gaussian_hm)
        # save_path = '/media/ps/lys/CBCT_tooth/data/test_hm'
        # sitk.WriteImage(new_mask, os.path.join(save_path, case.split('/')[-1]))

        return gaussian_hm