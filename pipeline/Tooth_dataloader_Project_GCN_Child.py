import os.path

import numpy as np
import module.common_module as cm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
from other_utils import Resample
from torchvision import transforms, utils
import module.transform as trans
import torch
import SimpleITK as sitk


def Toothdata(test_dir, shape, num_class = 1):
    test_imgList = sorted(glob(test_dir+'/*.nii.gz'))
    test_img_list = test_imgList


    # Build Dataset Class
    class ToothDataset(Dataset):

        def __init__(self, img_list, shape, num_classes,transform=None, test = False):
            self.img_list = img_list
            self.transform = transform
            self.shape = shape
            self.test = test
            self.num_classes = num_classes

        def __len__(self):
            return int(len(self.img_list))

        def __getitem__(self, idx):
            case_name = self.img_list[idx]
            image = sitk.ReadImage(self.img_list[idx])
            image = sitk.GetArrayFromImage(image)
            image = image.astype(np.float32)

            image, _ = self.pad(image, image, self.shape)

            image = image[np.newaxis, :, :, :]
            sample = {'case_name': case_name, 'image': image}
            return sample

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


    testDataset = ToothDataset(test_img_list,
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
    dataset_sizes = {'test': len(testDataset)}

    modelDataLoader = {'test': DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4)}

    return device, dataset_sizes, modelDataLoader