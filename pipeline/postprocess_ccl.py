import os
from os.path import join
import numpy as np
import torch
import SimpleITK as sitk
from other_utils import Resample,quadrant_locate,quadrant_merge,connected_domain_filter_v2

predict_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child_multi_center/Center1_sup_2/predict/tooth_prediction'
label_path = r'//media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child_multi_center/Center1_sup_2/label_cls'
predict_path_ccl = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child_multi_center/Center1_sup_2/tooth_prediction_post'
label_ccl_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child_multi_center/Center1_sup_2/label_cls_post'

if not os.path.exists(predict_path_ccl):
    os.mkdir(predict_path_ccl)

if not os.path.exists(label_ccl_path):
    os.mkdir(label_ccl_path)

for case in os.listdir(predict_path):
    print(case)
    mask = sitk.ReadImage(os.path.join(predict_path, case))
    mask_array = sitk.GetArrayFromImage(mask)
    Origin = mask.GetOrigin()
    Spacing = mask.GetSpacing()
    Direction = mask.GetDirection()
    idx = [i for i in np.unique(mask_array) if i != 0]
    pred_mask_ccl = np.zeros_like(mask_array)

    for ii in idx:
        temp_array = np.zeros_like(mask_array)
        temp_array[mask_array == ii] = 1
        temp_array = connected_domain_filter_v2(temp_array)
        pred_mask_ccl[temp_array > 0] = ii

    res_mask = sitk.GetImageFromArray(pred_mask_ccl)
    res_mask.SetOrigin(Origin)
    res_mask.SetSpacing(Spacing)
    res_mask.SetDirection(Direction)
    sitk.WriteImage(res_mask, os.path.join(predict_path_ccl, case))

    label = sitk.ReadImage(os.path.join(label_path, case))
    label_array = sitk.GetArrayFromImage(label)
    d_list = [9,10,11,12,13,23,24,25,26,27,51,52,53,54,55,37,38,39,40,41]
    label_idx = [i for i in np.unique(label_array) if i in idx and i not in d_list]
    for jj in label_idx:
        temp_array = np.zeros_like(mask_array)
        temp_array[mask_array == jj] = 1

        temp_array2 = np.zeros_like(mask_array)
        temp_array2[label_array== jj] = 1

        label_array[label_array == jj] = 0

        temp_array = temp_array * temp_array2
        label_array[temp_array>0] = jj
    res_mask2 = sitk.GetImageFromArray(label_array)
    res_mask2.SetOrigin(Origin)
    res_mask2.SetSpacing(Spacing)
    res_mask2.SetDirection(Direction)
    sitk.WriteImage(res_mask2, os.path.join(label_ccl_path, case))


