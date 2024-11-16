import os
import SimpleITK as sitk
import numpy as np

def Adjacent_Matrix_Generation(label,cls_num):
    # cls_num = len(np.unique(label))
    # print(cls_num)
    Adjacent_Matrix = np.zeros((cls_num, cls_num))
    LabelArr = np.copy(label)
    LabelArr[LabelArr == cls_num - 1] = 0
    cls_list = np.unique(LabelArr)
    cls_list = cls_list.tolist()
    cls_list.remove(0)
    # print((cls_list))
    for x_cls in cls_list:
        for y_cls in cls_list:
            x_cls = int(x_cls)
            y_cls = int(y_cls)
            if x_cls == y_cls:
                Adjacent_Matrix[x_cls][y_cls] = 1
                continue

            x_mask_array = np.zeros_like(LabelArr)
            y_mask_array = np.zeros_like(LabelArr)
            x_mask_array[LabelArr == x_cls] = 1
            y_mask_array[LabelArr == y_cls] = 1

            for i in range(10):
                x_mask = sitk.DilateObjectMorphology(sitk.GetImageFromArray(x_mask_array), (1, 1, 1), sitk.sitkBall,
                                                     int(1))
                y_mask = sitk.DilateObjectMorphology(sitk.GetImageFromArray(y_mask_array), (1, 1, 1), sitk.sitkBall,
                                                     int(1))
                x_mask_array = sitk.GetArrayFromImage(x_mask)
                y_mask_array = sitk.GetArrayFromImage(y_mask)
                union_set = sitk.GetArrayFromImage(x_mask) + sitk.GetArrayFromImage(y_mask)
                # sitk.WriteImage(sitk.GetImageFromArray(union_set),r'F:\adult_tooth\data\Quadrant\train\temp' + '\\' + str(x_cls)+str(y_cls)+case)
                if 2 in np.unique(union_set):
                    break

            count_array = np.copy(LabelArr)
            count_array[LabelArr == x_cls] = 0
            count_array[LabelArr == y_cls] = 0
            count_array[count_array != 0] = 1
            union_set[union_set > 0] = 1

            add_array = union_set + count_array
            # print(np.unique(count_array),np.unique(count_array + union_set),np.sum(add_array[add_array == 2]))
            if np.sum(add_array[add_array == 2]) < 500:
                # print(x_cls,y_cls)
                Adjacent_Matrix[x_cls][y_cls] = 1
                Adjacent_Matrix[y_cls][x_cls] = 1
    # print(Adjacent_Matrix,Adjacent_Matrix.shape)
    return Adjacent_Matrix,cls_num