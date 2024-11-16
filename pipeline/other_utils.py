import os
import SimpleITK as sitk
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math

def check_and_create_path(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

def Resample(Image, NewSpacing, Label, Size = None):
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)
    if not Size:
        NewSize = [int(Array.shape[2] * Spacing[0] / NewSpacing[0]), int(Array.shape[1] * Spacing[1] / NewSpacing[1]),
               int(Array.shape[0] * Spacing[2] / NewSpacing[2])]
    else:
        NewSize = Size
    # print(NewSize)
    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    if Label:
        Resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        Resample.SetInterpolator(sitk.sitkLinear)
    Resample.SetOutputSpacing(NewSpacing)

    NewImage = Resample.Execute(Image)

    return NewImage

def Normalize(Image, LowerBound, UpperBound):
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)

    Array[Array < LowerBound] = LowerBound
    Array[Array > UpperBound] = UpperBound
    # Array = (Array  - np.mean(Array )) / np.std(Array )
    Array = (Array.astype(np.float64) - LowerBound) / (UpperBound - LowerBound)
    Array = (Array * 255).astype(np.uint8)
    Image = sitk.GetImageFromArray(Array)
    Image.SetSpacing(Spacing)
    Image.SetOrigin(Origin)
    Image.SetDirection(Direction)
    return Image

def Padding_Size_Adaption(data_dir,label_dir, nnunet_norm = False):
    shape_z = []
    shape_x = []
    shape_y = []
    voxels_all = []
    for img in os.listdir(data_dir):
        data = sitk.ReadImage(os.path.join(data_dir, img))
        data_array = sitk.GetArrayFromImage(data)
        if nnunet_norm:
            seg = sitk.ReadImage(os.path.join(label_dir, img))
            seg = sitk.GetArrayFromImage(seg)
            mask = seg > 0
            voxels = list(data_array[mask][::10])
            voxels_all += voxels
        shape_z.append(data_array.shape[0])
        shape_y.append(data_array.shape[1])
        shape_x.append(data_array.shape[2])
    padding_size = [(max(shape_z) // 16 + 1) * 16, (max(shape_y) // 16 + 1) * 16, (max(shape_x) // 16 + 1) * 16]
    return padding_size,voxels_all

def Adjacent_Matrix_Generation(label,cls_num):
    # cls_num = len(np.unique(label))
    # print(cls_num)
    Adjacent_Matrix = np.zeros((cls_num, cls_num))
    LabelArr = label
    LabelArr[LabelArr == np.unique(label)[-1]] = 0
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

def connected_domain_filter(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask = sitk.GetImageFromArray(itk_mask)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    # print('111', num_connected_label)
    area_thresh = 10
    np_output_mask = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output_mask)
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_thresh:
            res_mask[np_output_mask == i] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())
    return res_itk

def connected_domain_filter_v2(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask[itk_mask > 0] = 1
    itk_mask = sitk.GetImageFromArray(itk_mask)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    # print('111', num_connected_label)
    area_thresh = 100
    np_output_mask = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output_mask)
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_thresh:
            res_mask[np_output_mask == i] = 1

    return res_mask

def error_connected_select(mask_array):
    mask_voxel_coords = np.where(mask_array == 1)
    # print(i)
    # print(mask_voxel_coords)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    center_z = int((minzidx + maxzidx) / 2)
    center_x = int((minxidx + maxxidx) / 2)
    center_y = int((minyidx + maxyidx) / 2)
    diameter_z = maxzidx - minzidx
    diameter_x = maxxidx - minxidx
    diameter_y = maxyidx - minyidx

    for w in [60]:
        flag = 0
        # bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
        minzidx = int(center_z - 0.5 * diameter_z * w / 100)
        maxzidx = int(center_z + 0.5 * diameter_z * w / 100)
        if minzidx < 0:
            minzidx = 0
        if maxzidx > mask_array.shape[0] - 1:
            maxzidx = mask_array.shape[0] - 1
        minxidx = int(center_x - 0.5 * diameter_x * w / 100)
        maxxidx = int(center_x + 0.5 * diameter_x * w / 100)
        if minxidx < 0:
            minxidx = 0
        if maxxidx > mask_array.shape[1] - 1:
            maxxidx = mask_array.shape[1] - 1
        minyidx = int(center_y - 0.5 * diameter_y * w / 100)
        maxyidx = int(center_y + 0.5 * diameter_y * w / 100)
        if minyidx < 0:
            minyidx = 0
        if maxyidx > mask_array.shape[2] - 1:
            maxyidx = mask_array.shape[2] - 1
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return resizer

def connected_domain_locate(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    instance_num = np.unique(itk_mask)
    mask_array = np.copy(itk_mask)
    # print(np.unique(itk_mask))
    box_container = []
    for i in instance_num:
        if i == 0:
            continue
        itk_copy = np.copy(itk_mask)
        itk_copy[itk_copy != i] = 0
        itk_copy[itk_copy == i] = 1
        resizer = error_connected_select(itk_copy)
        # if np.sum(itk_copy)<500:
        #     if len(np.unique(itk_mask[resizer])) != 2 or np.sum(itk_copy) < 10:
        #         continue
        # print(i,np.sum(itk_copy))
        itk = connected_domain_filter(itk_copy)
        itk_array = sitk.GetArrayFromImage(itk)
        if len(np.unique(itk_array)) == 1:
            continue
        mask_voxel_coords = np.where(itk_array == 1)
        # print(i)
        # print(mask_voxel_coords)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        center_z = int((minzidx + maxzidx) / 2)
        center_x = int((minxidx + maxxidx) / 2)
        center_y = int((minyidx + maxyidx) / 2)
        diameter_z = maxzidx - minzidx
        diameter_x = maxxidx - minxidx
        diameter_y = maxyidx - minyidx

        for w in [100]:
            flag = 0
            # bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
            minzidx = int(center_z - 0.5 * diameter_z * w / 100)
            maxzidx = int(center_z + 0.5 * diameter_z * w / 100)
            if minzidx < 0:
                minzidx = 0
            if maxzidx > itk_copy.shape[0] - 1:
                maxzidx = itk_copy.shape[0] - 1
            minxidx = int(center_x - 0.5 * diameter_x * w / 100)
            maxxidx = int(center_x + 0.5 * diameter_x * w / 100)
            if minxidx < 0:
                minxidx = 0
            if maxxidx > itk_copy.shape[1] - 1:
                maxxidx = itk_copy.shape[1] - 1
            minyidx = int(center_y - 0.5 * diameter_y * w / 100)
            maxyidx = int(center_y + 0.5 * diameter_y * w / 100)
            if minyidx < 0:
                minyidx = 0
            if maxyidx > itk_copy.shape[2] - 1:
                maxyidx = itk_copy.shape[2] - 1
            bbox = [minzidx, maxzidx, minxidx, maxxidx, minyidx,  maxyidx]
        box_container.append(bbox)
    print('boxes_num:', len(box_container))
    return box_container

def iou_criterion(bbox1, bbox2):
    """iou of two 3d bboxes. bbox: [z,y,x,dz,dy,dx]"""

    # zmin, zmax, ymin, ymax, xmin, xmax
    # bbox1 = [bbox1[0] - bbox1[3] / 2,
    #          bbox1[0] + bbox1[3] / 2,
    #          bbox1[1] - bbox1[4] / 2,
    #          bbox1[1] + bbox1[4] / 2,
    #          bbox1[2] - bbox1[5] / 2,
    #          bbox1[2] + bbox1[5] / 2]
    # bbox2 = [bbox2[0] - bbox2[3] / 2,
    #          bbox2[0] + bbox2[3] / 2,
    #          bbox2[1] - bbox2[4] / 2,
    #          bbox2[1] + bbox2[4] / 2,
    #          bbox2[2] - bbox2[5] / 2,
    #          bbox2[2] + bbox2[5] / 2]

    # Intersection bbox and volume.
    int_zmin = np.maximum(bbox1[0], bbox2[0])
    int_zmax = np.minimum(bbox1[1], bbox2[1])
    int_ymin = np.maximum(bbox1[2], bbox2[2])
    int_ymax = np.minimum(bbox1[3], bbox2[3])
    int_xmin = np.maximum(bbox1[4], bbox2[4])
    int_xmax = np.minimum(bbox1[5], bbox2[5])

    int_z = np.maximum(int_zmax - int_zmin, 0.0)
    int_y = np.maximum(int_ymax - int_ymin, 0.0)
    int_x = np.maximum(int_xmax - int_xmin, 0.0)

    int_vol = int_z * int_y * int_x

    vol1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2]) * (bbox1[5] - bbox1[4])
    vol2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2]) * (bbox2[5] - bbox2[4])

    union = vol1 + vol2 - int_vol
    iou = float(int_vol / union)
    return iou

def max_connected_domain(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask = sitk.GetImageFromArray(itk_mask)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    print('num:', num_connected_label)
    area_thresh = 10 #350 10 #300 #80 #15
    np_output_mask = sitk.GetArrayFromImage(output_mask)

    res_mask = np.zeros_like(np_output_mask)
    Shape = res_mask.shape
    # print(Shape)
    map = np.zeros_like(np_output_mask)
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        center_point = lss_filter.GetCentroid(i)
        # print(area)
        if area > area_thresh:
            res_mask[np_output_mask == i] = i
            bbox = [[center_point[2]-4, center_point[2]+4], [center_point[1]-10, center_point[1]+10], [center_point[0]-10, center_point[0]+10]]
            for idxx in range(3):
                if bbox[idxx][0] < 0:
                    bbox[idxx][0] = 0
                if bbox[idxx][1] > Shape[idxx]-1:
                    bbox[idxx][1] = Shape[idxx]-1
                bbox[idxx][0] = int(bbox[idxx][0])
                bbox[idxx][1] = int(bbox[idxx][1])
            resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
            # print(resizer)
            map[resizer] = i
        else:
            print('may not tooth', np.where(np_output_mask == i),area)
            if map[int(center_point[2]),int(center_point[1]),int(center_point[0])] == 0 and area > 3:
                # res_mask[np_output_mask == i] = i
                # print('still tooth')
                bbox = [[center_point[2]-4, center_point[2]+4], [center_point[1]-10, center_point[1]+10], [center_point[0]-10, center_point[0]+10]]
                for idxx in range(3):
                    if bbox[idxx][0] < 0:
                        bbox[idxx][0] = 0
                    if bbox[idxx][1] > Shape[idxx] - 1:
                        bbox[idxx][1] = Shape[idxx] - 1
                    bbox[idxx][0] = int(bbox[idxx][0])
                    bbox[idxx][1] = int(bbox[idxx][1])
                resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
                print(resizer)
                map[resizer] = i
    res_itk = sitk.GetImageFromArray(res_mask)
    map = sitk.GetImageFromArray(map)
    return res_itk,map

def split_label(mask_array):
    img , _= max_connected_domain(mask_array)
    return sitk.GetArrayFromImage(img)

def quadrant_locate(data_dir,quadrant_dir,save_resizer_path,save_quadrant_crop_path):
    if not os.path.exists(save_resizer_path):
        os.makedirs(save_resizer_path)
    if not os.path.exists(save_quadrant_crop_path):
        os.makedirs(save_quadrant_crop_path)
    for case in os.listdir(quadrant_dir):
        data = sitk.ReadImage(os.path.join(data_dir,case))
        data_array = sitk.GetArrayFromImage(data)

        mask_quadrant = sitk.ReadImage(os.path.join(quadrant_dir, case))
        mask_quadrant_array = sitk.GetArrayFromImage(mask_quadrant)

        resizer_dict = dict()


        for i in range(1,5):
            mask_voxel_coords = np.where(mask_quadrant_array == i)

            # print(mask_voxel_coords)
            minzidx = int(np.min(mask_voxel_coords[0]))
            maxzidx = int(np.max(mask_voxel_coords[0])) + 1
            minxidx = int(np.min(mask_voxel_coords[1]))
            maxxidx = int(np.max(mask_voxel_coords[1])) + 1
            minyidx = int(np.min(mask_voxel_coords[2]))
            maxyidx = int(np.max(mask_voxel_coords[2])) + 1
            center_z = int((minzidx + maxzidx) / 2)
            center_x = int((minxidx + maxxidx) / 2)
            center_y = int((minyidx + maxyidx) / 2)
            diameter_z = maxzidx - minzidx
            diameter_x = maxxidx - minxidx
            diameter_y = maxyidx - minyidx
            for w in [100]:
                flag = 0
                # bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
                minzidx = int(center_z - 0.5 * diameter_z * w / 100) - 2
                maxzidx = int(center_z + 0.5 * diameter_z * w / 100) + 2
                if minzidx < 0:
                    minzidx = 0
                if maxzidx > mask_quadrant_array.shape[0] - 1:
                    maxzidx = mask_quadrant_array.shape[0] - 1
                minxidx = int(center_x - 0.5 * diameter_x * w / 100) - 2
                maxxidx = int(center_x + 0.5 * diameter_x * w / 100) + 2
                if minxidx < 0:
                    minxidx = 0
                if maxxidx > mask_quadrant_array.shape[1] - 1:
                    maxxidx = mask_quadrant_array.shape[1] - 1
                minyidx = int(center_y - 0.5 * diameter_y * w / 100) - 2
                maxyidx = int(center_y + 0.5 * diameter_y * w / 100) + 2
                if minyidx < 0:
                    minyidx = 0
                if maxyidx > mask_quadrant_array.shape[2] - 1:
                    maxyidx = mask_quadrant_array.shape[2] - 1
                bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

                # resizer = (slice(bbox[0][0]-2, bbox[0][1]+2), slice(bbox[1][0]-2, bbox[1][1]+2), slice(bbox[2][0]-2, bbox[2][1]+2))
                resizer= (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
                print(case.replace('.nii', '_{}.nii'.format(str(i))), resizer)
                resizer_dict[case.replace('.nii', '_{}.nii'.format(str(i)))] = resizer

            data_crop_array = data_array[resizer]

            data_crop = sitk.GetImageFromArray(data_crop_array)
            data_crop.SetOrigin(data.GetOrigin())
            data_crop.SetSpacing(data.GetSpacing())
            data_crop.SetDirection(data.GetDirection())
            sitk.WriteImage(data_crop, os.path.join(save_quadrant_crop_path, case.replace('.nii', '_{}.nii'.format(str(i)))))
        np.save(os.path.join(save_resizer_path, case.replace('.nii.gz', '.npy')), resizer_dict)


def quadrant_merge(image_path,resample_path,quadrant_mask_path,save_test_resizer_path,mask_path):
    for case in os.listdir(save_test_resizer_path):
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        # if case not in ['Cui Jia__2016_12_12.npy']:
        #     continue
        print(case)
        directions = np.load(os.path.join(save_test_resizer_path, case), allow_pickle=True)
        result = dict(directions.tolist())
        image = sitk.ReadImage(os.path.join(image_path, case.replace('.npy', '.nii.gz')))

        image_resample = sitk.ReadImage(os.path.join(resample_path, case.replace('.npy', '.nii.gz')))
        Origin = image_resample.GetOrigin()
        Spacing = image_resample.GetSpacing()
        Direction = image_resample.GetDirection()
        image_array = sitk.GetArrayFromImage(image_resample)

        mask_copy_array = np.zeros_like(image_array)

        # print(result[case.replace('.npy','_1.nii.gz')])
        for idx, (key, value) in enumerate(dict(result).items()):
            print(idx, key, value)
            # if key not in ['Cui Jia__2016_12_12_3.nii.gz','Cui Jia__2016_12_12_1.nii.gz']:
            #     continue
            mask_array = np.zeros_like(image_array)
            quadrant_mask = sitk.ReadImage(os.path.join(quadrant_mask_path, key))
            quadrant_mask_array = sitk.GetArrayFromImage(quadrant_mask)
            quadrant_mask_array[quadrant_mask_array == 15] = 0
            quadrant_mask_array_copy = np.zeros_like(quadrant_mask_array)

            for i in np.unique(quadrant_mask_array):
                if i == 0:
                    continue
                quadrant_mask_array_copy[quadrant_mask_array == i] = idx * 14 + i
                # quadrant_mask_array_copy[quadrant_mask_array == i] = i

            # quadrant_mask_array_copy[quadrant_gt_array == 15] = 0
            mask_array[value] = quadrant_mask_array_copy
            mask_copy_array += mask_array
            # print(np.unique(mask_copy_array))

        mask = sitk.GetImageFromArray(mask_copy_array)

        if image.GetSpacing() != image_resample.GetSpacing():
            mask.SetDirection(Direction)
            mask.SetOrigin(Origin)
            mask.SetSpacing(Spacing)
            mask = Resample(mask, image.GetSpacing(), True, image.GetSize())

        else:
            mask.SetDirection(Direction)
            mask.SetOrigin(Origin)
            mask.SetSpacing(Spacing)
        sitk.WriteImage(mask, os.path.join(mask_path, case.replace('.npy', '.nii.gz')))

