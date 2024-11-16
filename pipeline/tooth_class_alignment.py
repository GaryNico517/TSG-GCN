import os
from os.path import join
import numpy as np
import torch
import SimpleITK as sitk

mask_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center2/tooth_prediction'

predict_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center2/predict/tooth_prediction'

save_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center2/label_cls'
for case in [i for i in os.listdir(mask_path) if i not in os.listdir(save_path)]:
    print(case)
    mask = sitk.ReadImage(os.path.join(mask_path,case))
    mask_array = sitk.GetArrayFromImage(mask)
    Origin = mask.GetOrigin()
    Spacing = mask.GetSpacing()
    Direction = mask.GetDirection()
    predict = sitk.ReadImage(os.path.join(predict_path, case))
    predict_array = sitk.GetArrayFromImage(predict)

    res_mask_array = np.zeros_like(mask_array)

    idx_list = [i for i in np.unique(mask_array) if i != 0]
    for idx in np.unique(idx_list):
        mask_temp_array = np.zeros_like(mask_array)
        mask_temp_array[mask_array == idx] = 1
        contain_array = predict_array * mask_temp_array
        idx_predict_list = [i for i in np.unique(contain_array) if i != 0]
        print('Containing',idx_predict_list)
        if len(idx_predict_list) == 0:
            res_mask_array[mask_temp_array == 1] = idx
            print(idx, '-->', idx,'-->Unfound')
            continue
        else:
            max_idx = 0
            max_cls_count = 0
            for jj in np.unique(idx_predict_list):
                projection_copy = np.zeros_like(contain_array)
                projection_copy[contain_array == jj] = 1
                cls_count = np.sum(projection_copy)
                if cls_count > max_cls_count:
                    max_cls_count = cls_count
                    max_idx = jj
            print(idx, '-->', max_idx)
            res_mask_array[mask_temp_array == 1] = max_idx

    res_mask = sitk.GetImageFromArray(res_mask_array)
    res_mask.SetOrigin(Origin)
    res_mask.SetSpacing(Spacing)
    res_mask.SetDirection(Direction)
    sitk.WriteImage(res_mask,os.path.join(save_path,case))

