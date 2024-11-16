# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
import torch
import SimpleITK as sitk
from unet3d import UNet3D
from other_utils import Resample,Normalize,quadrant_locate,quadrant_merge,connected_domain_filter_v2
from Quadrant_tooth_parsing import quadrant_tooth_parsing
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
import time
import argparse

def predict_quadrant(model, save_path, img_path, model_name,num_class,patch_size):
    print("Predict test data")
    model.eval()
    image_filenames = [x for x in os.listdir(img_path)]

    print(image_filenames)
    Low_spacing = [1,1,1]
    if not os.path.exists(join(save_path, 'Quadrant_label')):
        os.makedirs(join(save_path, 'Quadrant_label'))

    for imagename in image_filenames:
        print(imagename)
        image_1 = sitk.ReadImage(join(img_path, imagename))
        if Low_spacing:
            image = Resample(image_1, Low_spacing, False)
        else:
            image = image_1
        data = sitk.GetArrayFromImage(image)

        # LowerBound, UpperBound = -750, 3000
        # data = np.clip(data, LowerBound, UpperBound)  # -200, 600    -100, 300   -150, 400
        # data = (data.astype(np.float64) - LowerBound) / (UpperBound - LowerBound)

        # data = np.clip(data, -1000, 2000)  # -200, 600    -100, 300   -150, 400
        # mean = np.mean(train_voxels_all)
        # std = np.std(train_voxels_all)
        # percentile_99_5 = np.percentile(train_voxels_all, 99.5)
        # percentile_00_5 = np.percentile(train_voxels_all, 00.5)
        #
        # data = np.clip(data, percentile_00_5, percentile_99_5)
        # data = (data - mean) / std
        # if np.std(data) != .0:
        #     data = (data - np.mean(data)) / np.std(data)
        # data = (data * 255).astype(np.uint8)
        data = data.astype(np.float32)
        scaler = GradScaler()
        amp = True
        # padding
        padding_z = (16 - data.shape[0] % 16) % 16
        data = np.pad(data, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        padding_y = (16 - data.shape[1] % 16) % 16
        data = np.pad(data, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        padding_x = (16 - data.shape[2] % 16) % 16
        data = np.pad(data, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

        image = data[np.newaxis, np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        cut_size = np.array(patch_size)
        cut_step = cut_size // 2
        if num_class == 1:
            img = image
            patch_number = np.ceil((np.array(img.shape[2:]) - cut_size) / cut_step).astype(int) + 1
            out = np.zeros(img.shape[2:])
            pred_count = np.zeros(img.shape[2:])
            for i_z in range(patch_number[0]):
                for i_y in range(patch_number[1]):
                    for i_x in range(patch_number[2]):
                        z_start = i_z * cut_step[0]
                        z_end = min(i_z * cut_step[0] + cut_size[0], img.shape[2])
                        y_start = i_y * cut_step[1]
                        y_end = min(i_y * cut_step[1] + cut_size[1], img.shape[3])
                        x_start = i_x * cut_step[2]
                        x_end = min(i_x * cut_step[2] + cut_size[2], img.shape[4])
                        # print(start_point, end_point)
                        img_patch = img[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                        img_patch.requires_grad_(requires_grad=False)
                        img_patch = img_patch.cuda()

                        with autocast(enabled=amp):
                            with torch.no_grad():
                                out_patch = net_S(img_patch)  # N,C,D,H,W
                                out_patch = F.softmax(out_patch, dim=1)

                        out_patch = out_patch.squeeze().cpu().detach().numpy()
                        out[z_start:z_end, y_start:y_end, x_start:x_end] += out_patch
                        pred_count[z_start:z_end, y_start:y_end, x_start:x_end] += 1
                        del img_patch, out_patch
            out = out / pred_count
            pred_mask = (out > 0.5).astype(np.uint8)
        else:
            patch_number = []
            for idx in range(2,5):
                count = image.shape[idx] - cut_size[idx-2]
                if count < 0:
                    count = 0
                patch_num = np.ceil(count / cut_step[idx-2]).astype(int) + 1
                patch_number.append(patch_num)
            out = np.zeros([num_class,image.shape[2:][0],image.shape[2:][1],image.shape[2:][2]])
            pred_count = np.zeros([num_class,image.shape[2:][0],image.shape[2:][1],image.shape[2:][2]])
            for i_z in range(patch_number[0]):
                for i_y in range(patch_number[1]):
                    for i_x in range(patch_number[2]):
                        z_start = i_z * cut_step[0]
                        z_end = min(i_z * cut_step[0] + cut_size[0], image.shape[2])
                        y_start = i_y * cut_step[1]
                        y_end = min(i_y * cut_step[1] + cut_size[1], image.shape[3])
                        x_start = i_x * cut_step[2]
                        x_end = min(i_x * cut_step[2] + cut_size[2], image.shape[4])
                        # print(start_point, end_point)
                        img_patch = image[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                        img_patch.requires_grad_(requires_grad=False)
                        img_patch = img_patch.cuda()
                        with autocast(enabled=amp):
                            with torch.no_grad():
                                out_patch = net_S(img_patch)  # N,C,D,H,W
                                out_patch = F.softmax(out_patch, dim=1)
                        out_patch = out_patch.squeeze().cpu().detach().numpy()
                        for i in range(num_class):
                            out[i][z_start:z_end, y_start:y_end, x_start:x_end] += out_patch[i]
                            pred_count[i][z_start:z_end, y_start:y_end, x_start:x_end] += 1
                        del img_patch, out_patch
            out = (out / pred_count).squeeze()
            # print(out.shape)
            pred_mask = np.argmax(out, axis=0).astype(np.uint8)
        print(np.unique(pred_mask))
        pred_mask = pred_mask[padding_z // 2: pred_mask.shape[0] - (padding_z - padding_z // 2),
                     padding_y // 2: pred_mask.shape[1] - (padding_y - padding_y // 2),
                     padding_x // 2: pred_mask.shape[2] - (padding_x - padding_x // 2)]

        pred_mask_ccl = np.zeros_like(pred_mask)

        for ii in range(1,num_class):
            temp_array = np.zeros_like(pred_mask)
            temp_array[pred_mask == ii] = 1
            temp_array = connected_domain_filter_v2(temp_array)
            pred_mask_ccl[temp_array > 0] = ii

        out_pred = sitk.GetImageFromArray(pred_mask_ccl)
        if Low_spacing:
            out_pred = Resample(out_pred, image_1.GetSpacing(), True, image_1.GetSize())
        else:
            out_pred.SetSpacing(image_1.GetSpacing())
            out_pred.SetDirection(image_1.GetDirection())
            out_pred.SetOrigin(image_1.GetOrigin())

        sitk.WriteImage(out_pred, join(save_path,'Quadrant_label', imagename.replace('_0000','')))

def Data_Resample(test_dir,resample_dir,spacing = (0.4000000059604645, 0.4000000059604645, 0.4000000059604645)):
    print('Data Preprocessing...')
    if not os.path.exists(resample_dir):
        os.mkdir(resample_dir)
    for case in os.listdir(test_dir):
        data = sitk.ReadImage(os.path.join(test_dir,case))
        if data.GetSpacing() != spacing:
            data = Resample(data, spacing, False)
            data = Normalize(data, -750, 3000)
        sitk.WriteImage(data,os.path.join(resample_dir,case))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--pred_dir_QuadrantSeg', type=str, required=False,
                        default=r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center_public/Quadrant')
    parser.add_argument("--checkpoint_dir_QuadrantSeg", type=str, required=False,
                        default='../result_Project/Adult_Tooth_quadrant/seed1/model/3DUNet_Quadrant/save/best_model-300-0.9411,0.9452,0.9375.pth')
    parser.add_argument('--test_dir_QuadrantSeg', type=str, required=False,
                        default=r'/media/ps/lys_ssd/Project/Project_Multi_Child_TMI/data_GCN/adult_multi_center/Center_public/data')
    parser.add_argument('--checkpoint_dir_ToothSeg', type=str, required=False,
                        default=r'../result_Project/Adult_Tooth_quadrant/seed1/model/TSG-GCN/save/best_model-300-0.9283-0.9319-0.9250-0.6016-0.0000.pth')
    args = parser.parse_args()
    n_classes = 5
    crop_shape = (128,160,160) #(128, 128, 128)
    model_name = "3DUNet_Quadrant"
    pred_dir = args.pred_dir_QuadrantSeg
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    checkpoint_dir = args.checkpoint_dir_QuadrantSeg
    test_dir = args.test_dir_QuadrantSeg
    resample_test_dir = os.path.join(pred_dir, 'data_resampled')

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    since = time.time()
    # Data_Resample(test_dir,resample_test_dir,spacing=(0.5,0.5,0.5))
    data_time = time.time()
    net_S = UNet3D(in_channels=1, out_channels=n_classes, init_features=32).cuda()
    net_S.load_state_dict(torch.load(checkpoint_dir))

    predict_quadrant(net_S, pred_dir, resample_test_dir, model_name, num_class=n_classes,patch_size = crop_shape)
    predict_quadrant_time = time.time()
    quadrant_locate(resample_test_dir,os.path.join(pred_dir,'Quadrant_label'),os.path.join(pred_dir,'resizer_npy'),os.path.join(pred_dir,'Quadrant_data'))
    quadrant_locate_time = time.time()

    num_classes = 9
    shape = [96,160,160]
    checkpoint_dir_ToothSeg = args.checkpoint_dir_ToothSeg
    save_result_path = os.path.join(pred_dir,'Quadrant_tooth_results')
    quadrant_tooth_parsing(num_classes,shape,os.path.join(pred_dir,'Quadrant_data'),checkpoint_dir_ToothSeg,save_result_path)
    quadrant_tooth_parsing_time = time.time()
    quadrant_merge(test_dir,resample_test_dir,save_result_path,os.path.join(pred_dir,'resizer_npy'),os.path.join(pred_dir,'tooth_prediction'))
    quadrant_merge_time = time.time()
    time_elapsed = time.time() - since
    print('Data Preprocess in {:.0f}m {:.0f}s'.format((data_time-since) // 60, (data_time-since) % 60))
    print('Quadrant Prediction in {:.0f}m {:.0f}s'.format((predict_quadrant_time - data_time) // 60, (predict_quadrant_time - data_time) % 60))
    print('Quadrant Cut in {:.0f}m {:.0f}s'.format((quadrant_locate_time - predict_quadrant_time) // 60, (quadrant_locate_time - predict_quadrant_time) % 60))
    print('Quadrant Tooth Segmentation in {:.0f}m {:.0f}s'.format((quadrant_tooth_parsing_time - quadrant_locate_time) // 60, (quadrant_tooth_parsing_time - quadrant_locate_time) % 60))
    print('Quadrant Merge in {:.0f}m {:.0f}s'.format((quadrant_merge_time - quadrant_tooth_parsing_time) // 60, (quadrant_merge_time - quadrant_tooth_parsing_time) % 60))

    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

