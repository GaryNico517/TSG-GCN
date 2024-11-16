import shutil
import sys
import os
import random
import json
import numpy
import SimpleITK as sitk
import numpy as np


mask_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center1/label_company'
resizer_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center1/predict/resizer_npy'

predict_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center1/predict/Quadrant_tooth_results'
save_quadrant_path = r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center1/predict/gt'

idx_list = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    [15,16,17,18,19,20,21,22,23,24,25,26,27,28],
    [29,30,31,32,33,34,35,36,37,38,39,40,41,42],
    [43,44,45,46,47,48,49,50,51,52,53,54,55,56]])

if not os.path.exists(save_quadrant_path):
    os.mkdir(save_quadrant_path)


for case in [ i for i in os.listdir(resizer_path) if i.replace('.npy','.nii.gz') in os.listdir(mask_path)]:
    print(case)
    directions = np.load(os.path.join(resizer_path,case),allow_pickle=True)
    result = dict(directions.tolist())
    mask = sitk.ReadImage(os.path.join(mask_path,case.replace('.npy','.nii.gz')))

    mask_array = sitk.GetArrayFromImage(mask)

    # print(result[case.replace('.npy','_1.nii.gz')])
    for idx,(key,value) in enumerate(dict(result).items()):
        print(idx,key,value)
        quadrant_mask = sitk.ReadImage(os.path.join(predict_path,key))
        Origin = quadrant_mask.GetOrigin()
        Spacing = quadrant_mask.GetSpacing()
        Direction = quadrant_mask.GetDirection()
        # quadrant_mask_array = sitk.GetArrayFromImage(quadrant_mask)

        quadrant_gt_array = np.copy(mask_array[value])
        label_idx = idx_list[idx]

        for kk in [j for j in np.unique(quadrant_gt_array) if j not in label_idx and j != 0]:
            quadrant_gt_array[quadrant_gt_array == kk] = 99


        for ii in label_idx:
            quadrant_gt_array[quadrant_gt_array == ii] = ii - 14 * idx

        quadrant_gt_array[quadrant_gt_array == 99] = 15



        quadrant_gt = sitk.GetImageFromArray(quadrant_gt_array)
        quadrant_gt.SetDirection(Direction)
        quadrant_gt.SetOrigin(Origin)
        quadrant_gt.SetSpacing(Spacing)
        sitk.WriteImage(quadrant_gt, os.path.join(save_quadrant_path, key))


