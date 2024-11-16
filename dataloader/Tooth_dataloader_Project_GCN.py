import os.path

import numpy as np
import module.common_module as cm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms, utils
import module.transform as trans
import torch
import SimpleITK as sitk
from dataloader.Adjacent_Matrix_Generation import Adjacent_Matrix_Generation
from dataloader.Distance_Map_Generation import Distance_Map_Generation
from module.DAML import AM_Generate_GT
import copy

def AM_Obtain(label_list,num_class,am_npy_dir):
    AM_list = []
    if not os.path.exists(am_npy_dir):
        os.mkdir(am_npy_dir)
    if len(os.listdir(am_npy_dir)) != len(label_list):
        print('Writing Adjacent Matrix File')
        for case in label_list:
            mask = sitk.ReadImage(case)
            mask = sitk.GetArrayFromImage(mask)
            case_AM, _ = Adjacent_Matrix_Generation(mask, num_class)
            AM_list.append(case_AM)
            # print('Writing: ',os.path.join(am_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')))
            np.save(os.path.join(am_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')),case_AM)
            # print(case_AM)
    else:
        print('Loading Adjacent Matrix File')
        for case in label_list:
            # print('Loading: ',os.path.join(am_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')))
            case_AM = np.load(os.path.join(am_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')))
            AM_list.append(case_AM)
    return AM_list

def DM_Obtain(label_list,dm_npy_dir):
    DM_list = []
    if not os.path.exists(dm_npy_dir):
        os.mkdir(dm_npy_dir)
    if len(os.listdir(dm_npy_dir)) != len(label_list):
        print('Writing Distance Map File')
        for case in label_list:
            mask = sitk.ReadImage(case)
            mask = sitk.GetArrayFromImage(mask)
            case_AM = Distance_Map_Generation(mask)
            DM_list.append(case_AM)
            # print('Writing: ',os.path.join(dm_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')))
            np.save(os.path.join(dm_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')),case_AM)
            # print(case_AM)
    else:
        print('Loading Distance Map File')
        for case in label_list:
            # print('Loading: ',os.path.join(dm_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')))
            case_AM = np.load(os.path.join(dm_npy_dir,case.split('/')[-1].replace('.nii.gz','.npy')))
            DM_list.append(case_AM)
    return DM_list

def AdultToothdata(data_seed, data_split, shape, num_class = 1):

    # Set random seed
    data_seed = data_seed
    np.random.seed(data_seed)

    # Create image list
    train_imgList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult/Quadrant/train/image/*.nii.gz'))
    train_maskList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult/Quadrant/train/label/*.nii.gz'))
    # train_npy_dir = r'/media/ps/lys/CBCT_tooth/Project_GCN/Quadrant/train/adjacent_matrix'
    # train_dmList = sorted(glob('/media/ps/lys/CBCT_tooth/data/train/3DUNet_Quadrant_low_res/predict_dismap/*.nii.gz'))
    # train_imgList = train_imgList[:6]
    # train_maskList = train_maskList[:6]
    # train_dmList = train_dmList[:6]
    # train_AMList = AM_Obtain(train_maskList,num_class,train_npy_dir)
    # train_DMList = DM_Obtain(train_maskList,train_dm_npy_dir)

    train_list = [list(pair) for pair in zip(train_imgList, train_maskList)]
    # train_list = train_list[:10]
    test_imgList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult/Quadrant/test/image/*.nii.gz'))
    test_maskList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult/Quadrant/test/label/*.nii.gz'))
    # test_npy_dir = r'/media/ps/lys/CBCT_tooth/Project_GCN/Quadrant/test/adjacent_matrix'
    # test_dmList = sorted(glob('/media/ps/lys/CBCT_tooth/data/test/3DUNet_Quadrant_low_res/predict_dismap/*.nii.gz'))
    # test_imgList = test_imgList[:6]
    # test_maskList = test_maskList[:6]
    # test_dmList = test_dmList[:6]
    # test_AMList = AM_Obtain(test_maskList, num_class,test_npy_dir)
    # test_DMList = DM_Obtain(test_maskList,test_dm_npy_dir)

    test_list = [list(pair) for pair in zip(test_imgList, test_maskList)]

    train_labeled_img_list = []
    train_labeled_mask_list = []
    train_labeled_AM_list = []
    train_labeled_DM_list = []

    val_labeled_img_list = []
    val_labeled_mask_list = []
    val_labeled_AM_list = []
    val_labeled_DM_list = []

    test_img_list = []
    test_mask_list = []
    test_labeled_AM_list = []
    test_labeled_DM_list = []

    if data_split == 'Adult_Tooth_quadrant':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(train_list)))
        val_labeled_img_list, val_labeled_mask_list= map(list, zip(*(test_list)))
        test_img_list, test_mask_list= map(list, zip(*(test_list)))


    # Reset random seed
    seed = random.randint(1, 9999999)
    np.random.seed(seed + 1)

    # Build Dataset Class
    class ToothDataset(Dataset):

        def __init__(self, img_list, mask_list, shape, num_classes,transform=None, test = False):
            self.img_list = img_list
            self.mask_list = mask_list
            # self.AM_list = AM_list
            # self.DM_list = DM_list
            self.transform = transform
            self.shape = shape
            self.test = test
            self.num_classes = num_classes

        def __len__(self):
            # assert len(self.img_list) == len(self.mask_list)
            return len(self.img_list)

        def __getitem__(self, idx):
            case_name = self.img_list[idx]
            image = sitk.ReadImage(self.img_list[idx])
            image = sitk.GetArrayFromImage(image)
            image = image.astype(np.float32)

            mask = sitk.ReadImage(self.mask_list[idx])
            mask = sitk.GetArrayFromImage(mask)
            mask = mask.astype(np.float32)
            mask[mask == self.num_classes] = 0
            # dis_map = sitk.ReadImage(self.DM_list[idx])
            # dis_map = sitk.GetArrayFromImage(dis_map)
            # dis_map = dis_map.astype(np.float32)

            # AM = self.AM_list[idx]

            if not self.test:

                image, mask = self.pad(image, mask, self.shape)
                # dis_map, _ = self.pad(dis_map, dis_map, self.shape)

                image = image[np.newaxis,  :, :, :]
                # dis_map = dis_map[np.newaxis,  :, :, :]
                # image = np.concatenate((image,dis_map),axis=0)

                # mask = mask[np.newaxis,  :, :, :]

                # AM = np.zeros((num_class,num_class))

                sample = {'image': image, 'mask': mask}
                if self.transform:
                    sample = self.transform(sample)
                image = sample.get("image")
                mask = sample.get("mask")


                # for cls_i in range(num_class):
                #     if cls_i not in np.unique(mask):
                #         AM[cls_i, :] = 0
                #         AM[:, cls_i] = 0
                #
                # AM_flatten = []
                # for y in range(AM.shape[0]):
                #     # print(AM[y][y:],len(AM_flatten))
                #     AM_flatten += AM[y][y:].tolist()
                #
                # AM_flatten = np.array(AM_flatten)


                # print(np.unique(mask))

                mask_for_distance = mask # wo stuff
                # mask_for_distance = copy.deepcopy(mask) #w stuff
                mask_for_distance[mask_for_distance == num_class] = 0

                mask_array_2 = np.copy(mask)
                mask_array_oh = self.to_categorical(mask_array_2, num_class)

                background = np.copy(mask_array_oh[0])
                mask_array_oh[0] = 1 - background

                mip_max_mask = np.max(mask_array_oh, axis=1)

                background_mip = np.copy(mip_max_mask[0])
                mip_max_mask[0] = 1 - background_mip
                # print(np.unique(mip_max_mask))
                distance_map = Distance_Map_Generation(mask_for_distance.numpy())
                distance_map = torch.tensor(distance_map, dtype=torch.float)
                mask_copy = torch.clone(mask)
                if self.num_classes != 1:
                    mask = self.to_categorical(mask, self.num_classes)

                AM = AM_Generate_GT(mip_max_mask)

                sample = {'case_name': case_name,'image': image, 'mask': mask,'mask_oc':mask_copy, 'am':AM,'mmip_mask':mip_max_mask, 'distance_map':distance_map}
                return sample
            else:
                image, mask = self.pad(image, mask, self.shape)
                # dis_map, _ = self.pad(dis_map, dis_map, self.shape)

                image = image[np.newaxis, :, :, :]
                # dis_map = dis_map[np.newaxis, :, :, :]

                mask[mask == num_class] = 0

                mask_array_2 = np.copy(mask)
                mask_array_oh = self.to_categorical(mask_array_2, num_class)

                background = np.copy(mask_array_oh[0])
                mask_array_oh[0] = 1 - background

                mip_max_mask = np.max(mask_array_oh, axis=1)

                background_mip = np.copy(mip_max_mask[0])
                mip_max_mask[0] = 1 - background_mip

                # image = np.concatenate((image, dis_map), axis=0)
                # AM_flatten = []
                # for y in range(AM.shape[0]):
                #     # print(AM[y][y:],len(AM_flatten))
                #     AM_flatten += AM[y][y:].tolist()

                # AM_flatten = np.array(AM_flatten)

                distance_map = Distance_Map_Generation(mask)
                mask_copy = np.copy(mask)
                if self.num_classes != 1:
                    mask = self.to_categorical(mask, self.num_classes)

                AM = AM_Generate_GT(mip_max_mask)

                sample = {'case_name': case_name, 'image': image, 'mask': mask,'mask_oc':mask_copy, 'am':AM,'mmip_mask':mip_max_mask, 'distance_map':distance_map}
                return sample

        # def pad(self, image, label, croped_shape):
        #     if image.shape[0] < croped_shape[0]:
        #         zero = np.zeros((croped_shape[0] - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
        #         image = np.concatenate([image, zero], axis=0)
        #         label = np.concatenate([label, zero], axis=0)
        #     if image.shape[1] < croped_shape[1]:
        #         zero = np.zeros((image.shape[0], croped_shape[1] - image.shape[1], image.shape[2]), dtype=np.float32)
        #         image = np.concatenate([image, zero], axis=1)
        #         label = np.concatenate([label, zero], axis=1)
        #     if image.shape[2] < croped_shape[2]:
        #         zero = np.zeros((image.shape[0], image.shape[1], croped_shape[2] - image.shape[2]), dtype=np.float32)
        #         image = np.concatenate([image, zero], axis=2)
        #         label = np.concatenate([label, zero], axis=2)
        #
        #     return image, label

        def pad(self, image, label, croped_shape):
            if image.shape[0] < croped_shape[0]:
                padding_z = croped_shape[0] - image.shape[0]
                image = np.pad(image, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
                label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
            if image.shape[1] < croped_shape[1]:
                padding_y = croped_shape[1] - image.shape[1]
                image = np.pad(image, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
                label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
            if image.shape[2] < croped_shape[2]:
                padding_x = croped_shape[2] - image.shape[2]
                image = np.pad(image, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
                label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

            return image, label

        def to_categorical(self, y, num_classes=None):
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

    # Iterating through the dataset
    trainLabeledDataset = ToothDataset(train_labeled_img_list, train_labeled_mask_list,
                                       shape,num_class,
                                     transform=transforms.Compose([
                                         trans.RandomCrop(tuple(shape)),
                                         # trans.Elastic(),
                                         trans.Flip(horizontal=True),
                                         trans.ToTensor()
                                     ])
                                     )


    valLabeledDataset = ToothDataset(val_labeled_img_list, val_labeled_mask_list,
                                     shape,num_class,
                                   transform=transforms.Compose([
                                       trans.Crop(tuple(shape)),
                                       trans.Flip(),
                                       trans.ToTensor()
                                   ])
                                   )

    testDataset = ToothDataset(test_img_list, test_mask_list,
                               shape,num_class,
                             transform=transforms.Compose([
                                 trans.ToTensor()
                             ]),
                            test = True
                             )

    # device_type = 'cpu'
    device_type = 'cuda'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    dataset_sizes = {'trainLabeled': len(trainLabeledDataset), 'val_labeled': len(valLabeledDataset), 'test': len(testDataset)}

    modelDataLoader = {'trainLabeled': DataLoader(trainLabeledDataset, batch_size=2, shuffle=True, num_workers=4),
                       'val_labeled': DataLoader(valLabeledDataset, batch_size=1, shuffle=False, num_workers=4),
                       'test': DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4)}

    return device, dataset_sizes, modelDataLoader

def ChildToothdata(data_seed, data_split, shape, num_class = 1):

    # Set random seed
    data_seed = data_seed
    np.random.seed(data_seed)

    # Create image list
    train_imgList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child/Quadrant_8_5_1/train/image/*.nii.gz'))
    train_maskList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child/Quadrant_8_5_1/train/label/*.nii.gz'))

    DA_cases_dir = '/media/ps/lys/CBCT_tooth/Project_Multi_Child/data_GCN/child/Quadrant_8_5_1/train/DA_cases.txt'
    DA_images_cases = []
    DA_labels_cases = []
    # for line in open(DA_cases_dir):
    #     DA_images_cases.append('/media/ps/lys/CBCT_tooth/Project_Multi_Child/data_GCN/child/Quadrant_8_5_1/train/image/'+str(line).replace('\n',''))
    #     DA_labels_cases.append('/media/ps/lys/CBCT_tooth/Project_Multi_Child/data_GCN/child/Quadrant_8_5_1/train/label/'+str(line).replace('\n',''))
    # train_imgList += DA_images_cases * 2
    # train_maskList += DA_labels_cases * 2

    # train_npy_dir = r'/media/ps/lys/CBCT_tooth/Project_GCN/Quadrant/train/adjacent_matrix'
    # train_dmList = sorted(glob('/media/ps/lys/CBCT_tooth/data/train/3DUNet_Quadrant_low_res/predict_dismap/*.nii.gz'))
    # train_imgList = train_imgList[:6]
    # train_maskList = train_maskList[:6]
    # train_dmList = train_dmList[:6]
    # train_AMList = AM_Obtain(train_maskList,num_class,train_npy_dir)
    # train_DMList = DM_Obtain(train_maskList,train_dm_npy_dir)

    train_list = [list(pair) for pair in zip(train_imgList, train_maskList)]
    # train_list = train_list[:10]
    test_imgList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child/Quadrant_8_5_1/test/image/*.nii.gz'))
    test_maskList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child/Quadrant_8_5_1/test/label/*.nii.gz'))

    # test_imgList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child_multi_center/Center2/predict/Quadrant_data/*.nii.gz'))
    # test_maskList = sorted(glob('/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/child_multi_center/Center2/predict/gt/*.nii.gz'))

    # test_npy_dir = r'/media/ps/lys/CBCT_tooth/Project_GCN/Quadrant/test/adjacent_matrix'
    # test_dmList = sorted(glob('/media/ps/lys/CBCT_tooth/data/test/3DUNet_Quadrant_low_res/predict_dismap/*.nii.gz'))
    # test_imgList = test_imgList[:6]
    # test_maskList = test_maskList[:6]
    # test_dmList = test_dmList[:6]
    # test_AMList = AM_Obtain(test_maskList, num_class,test_npy_dir)
    # test_DMList = DM_Obtain(test_maskList,test_dm_npy_dir)

    test_list = [list(pair) for pair in zip(test_imgList, test_maskList)]

    train_labeled_img_list = []
    train_labeled_mask_list = []
    train_labeled_AM_list = []
    train_labeled_DM_list = []

    val_labeled_img_list = []
    val_labeled_mask_list = []
    val_labeled_AM_list = []
    val_labeled_DM_list = []

    test_img_list = []
    test_mask_list = []
    test_labeled_AM_list = []
    test_labeled_DM_list = []

    if data_split == 'Child_Tooth_quadrant':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(train_list)))
        val_labeled_img_list, val_labeled_mask_list= map(list, zip(*(test_list)))
        test_img_list, test_mask_list= map(list, zip(*(test_list)))


    # Reset random seed
    seed = random.randint(1, 9999999)
    np.random.seed(seed + 1)

    # Build Dataset Class
    class ToothDataset(Dataset):

        def __init__(self, img_list, mask_list, shape, num_classes,iteration=1,transform=None, test = False):
            self.img_list = img_list
            self.mask_list = mask_list
            # self.AM_list = AM_list
            # self.DM_list = DM_list
            self.transform = transform
            self.shape = shape
            self.test = test
            self.num_classes = num_classes
            self.iteration = iteration

        def __len__(self):
            if not self.test:
                return int(len(self.img_list)) * self.iteration
            else:
                return int(len(self.img_list))

        def __getitem__(self, idx):
            case_name = self.img_list[idx]
            image = sitk.ReadImage(self.img_list[idx])
            image = sitk.GetArrayFromImage(image)
            image = image.astype(np.float32)

            mask = sitk.ReadImage(self.mask_list[idx])
            mask = sitk.GetArrayFromImage(mask)
            mask = mask.astype(np.float32)
            # mask[mask == self.num_classes] = 0

            # dis_map = sitk.ReadImage(self.DM_list[idx])
            # dis_map = sitk.GetArrayFromImage(dis_map)
            # dis_map = dis_map.astype(np.float32)

            # AM = self.AM_list[idx]

            if not self.test:
                image, mask = self.pad(image, mask, self.shape)
                # dis_map, _ = self.pad(dis_map, dis_map, self.shape)

                image = image[np.newaxis,  :, :, :]
                # dis_map = dis_map[np.newaxis,  :, :, :]
                # image = np.concatenate((image,dis_map),axis=0)

                # mask = mask[np.newaxis,  :, :, :]

                # AM = np.zeros((num_class,num_class))

                sample = {'image': image, 'mask': mask}
                if self.transform:
                    sample = self.transform(sample)
                image = sample.get("image")
                mask = sample.get("mask")


                # for cls_i in range(num_class):
                #     if cls_i not in np.unique(mask):
                #         AM[cls_i, :] = 0
                #         AM[:, cls_i] = 0
                #
                # AM_flatten = []
                # for y in range(AM.shape[0]):
                #     # print(AM[y][y:],len(AM_flatten))
                #     AM_flatten += AM[y][y:].tolist()
                #
                # AM_flatten = np.array(AM_flatten)


                # print(np.unique(mask))

                mask_for_distance = mask # wo stuff
                # mask_for_distance = copy.deepcopy(mask) #w stuff
                mask_for_distance[mask_for_distance == num_class] = 0

                mask_array_2 = np.copy(mask)
                mask_array_oh = self.to_categorical(mask_array_2, num_class) #恢复

                background = np.copy(mask_array_oh[0])
                mask_array_oh[0] = 1 - background

                mip_max_mask = np.max(mask_array_oh, axis=1)

                background_mip = np.copy(mip_max_mask[0])
                mip_max_mask[0] = 1 - background_mip #恢复

                # mip_mask = np.max(mask_array_2, axis=0)
                # mip_max_mask_2 = self.to_categorical(mip_mask, num_class)

                # print(np.unique(mip_max_mask))
                distance_map = Distance_Map_Generation(mask_for_distance.numpy())
                distance_map = torch.tensor(distance_map, dtype=torch.float)
                mask_copy = torch.clone(mask)
                if self.num_classes != 1:
                    mask = self.to_categorical(mask, self.num_classes)

                AM = AM_Generate_GT(mip_max_mask)

                sample = {'case_name': case_name,'image': image, 'mask': mask,'mask_oc':mask_copy, 'am':AM,'mmip_mask':mip_max_mask, 'distance_map':distance_map}
                return sample
            else:
                image, mask = self.pad(image, mask, self.shape)
                # dis_map, _ = self.pad(dis_map, dis_map, self.shape)

                image = image[np.newaxis, :, :, :]
                # dis_map = dis_map[np.newaxis, :, :, :]

                # mask[mask == num_class] = 0

                mask_array_2 = np.copy(mask)
                mask_array_oh = self.to_categorical(mask_array_2, num_class) #恢复

                background = np.copy(mask_array_oh[0])
                mask_array_oh[0] = 1 - background

                mip_max_mask = np.max(mask_array_oh, axis=1)

                background_mip = np.copy(mip_max_mask[0])
                mip_max_mask[0] = 1 - background_mip #恢复

                # mip_mask = np.max(mask_array_2, axis=0)
                # mip_max_mask_2 = self.to_categorical(mip_mask, num_class)

                # image = np.concatenate((image, dis_map), axis=0)
                # AM_flatten = []
                # for y in range(AM.shape[0]):
                #     # print(AM[y][y:],len(AM_flatten))
                #     AM_flatten += AM[y][y:].tolist()

                # AM_flatten = np.array(AM_flatten)

                distance_map = Distance_Map_Generation(mask)
                mask_copy = np.copy(mask)
                if self.num_classes != 1:
                    mask = self.to_categorical(mask, self.num_classes)

                AM = AM_Generate_GT(mip_max_mask)

                sample = {'case_name': case_name, 'image': image, 'mask': mask,'mask_oc':mask_copy, 'am':AM,'mmip_mask':mip_max_mask, 'distance_map':distance_map}
                return sample


        # def pad(self, image, label, croped_shape):
        #     if image.shape[0] < croped_shape[0]:
        #         zero = np.zeros((croped_shape[0] - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
        #         image = np.concatenate([image, zero], axis=0)
        #         label = np.concatenate([label, zero], axis=0)
        #     if image.shape[1] < croped_shape[1]:
        #         zero = np.zeros((image.shape[0], croped_shape[1] - image.shape[1], image.shape[2]), dtype=np.float32)
        #         image = np.concatenate([image, zero], axis=1)
        #         label = np.concatenate([label, zero], axis=1)
        #     if image.shape[2] < croped_shape[2]:
        #         zero = np.zeros((image.shape[0], image.shape[1], croped_shape[2] - image.shape[2]), dtype=np.float32)
        #         image = np.concatenate([image, zero], axis=2)
        #         label = np.concatenate([label, zero], axis=2)
        #
        #     return image, label

        def pad(self, image, label, croped_shape):
            if image.shape[0] < croped_shape[0]:
                padding_z = croped_shape[0] - image.shape[0]
                image = np.pad(image, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
                label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
            if image.shape[1] < croped_shape[1]:
                padding_y = croped_shape[1] - image.shape[1]
                image = np.pad(image, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
                label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
            if image.shape[2] < croped_shape[2]:
                padding_x = croped_shape[2] - image.shape[2]
                image = np.pad(image, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
                label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

            return image, label

        def to_categorical(self, y, num_classes=None):
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

    # Iterating through the dataset
    trainLabeledDataset = ToothDataset(train_labeled_img_list, train_labeled_mask_list,
                                       shape,num_class,
                                     transform=transforms.Compose([
                                         trans.RandomCrop(tuple(shape)),
                                         # trans.Elastic(),
                                         trans.Flip(horizontal=True),
                                         trans.ToTensor()
                                     ])
                                     )


    valLabeledDataset = ToothDataset(val_labeled_img_list, val_labeled_mask_list,
                                     shape,num_class,
                                   transform=transforms.Compose([
                                       trans.Crop(tuple(shape)),
                                       trans.Flip(),
                                       trans.ToTensor()
                                   ])
                                   )

    testDataset = ToothDataset(test_img_list, test_mask_list,
                               shape,num_class,
                             transform=transforms.Compose([
                                 trans.ToTensor()
                             ]),
                            test = True
                             )

    # device_type = 'cpu'
    device_type = 'cuda'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    dataset_sizes = {'trainLabeled': len(trainLabeledDataset), 'val_labeled': len(valLabeledDataset), 'test': len(testDataset)}

    modelDataLoader = {'trainLabeled': DataLoader(trainLabeledDataset, batch_size=2, shuffle=True, num_workers=4),
                       'val_labeled': DataLoader(valLabeledDataset, batch_size=1, shuffle=False, num_workers=4),
                       'test': DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4)}

    return device, dataset_sizes, modelDataLoader
