from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast,GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import module.common_module as cm
from module.dice_loss import DiceCoefficientLF,DiceLoss
from module.Huber_loss import MSE,Huber_loss
from module.loss_func import GeneralizedDiceLoss
from module.Crossentropy import crossentropy
from module.visualize_attention import visualize_Seg, visualize_Rec, visualize_loss
# from module.eval_GCN_slidingwindow import test_net_dice_w_2dproj
from module.loss_dict import HuberLoss,WeightedCE,cosine_similarity,Guassian_MSE,SSIM,FocalLoss
import os
from collections import defaultdict
from network import TSG_GCN,graph
import time
import copy
from tqdm import trange
import scipy.sparse as sp
import warnings
from dataloader.utils import CosineAnnealingLRWithRestarts,print_model_parm_nums
from module.DAML import AM_Generate_Pred,ThreeD_Seg_to_TwoD_Proj
import SimpleITK as sitk
from copy import deepcopy
from pipeline import Tooth_dataloader_Project_GCN_Child
from other_utils import Resample
from tqdm import tqdm
import skimage.measure


def network_training_epoch(model_load_path,save_result_path,device, data_sizes, modelDataLoader, num_class,shape, TSNE, GNN_model,SEG_AM_DM):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    val_dice = 0
    test_results = 0
    switch = {'trainL_encoder': True,
              'trainL_decoder_seg': True,
              'trainL_decoder_am': True,
              'trainL_decoder_dm': True}
    device = device
    dataset_sizes = data_sizes

    print('-' * 64)
    print('Tooth Parsing Start')

    base_features = 32
    AM_edge_num = int((num_class + 1) * num_class / 2)

    in_channels = 1
    model = TSG_GCN.GCN_v1(in_channels, num_class, AM_edge_num,base_features,Seg = SEG_AM_DM[0],AM = SEG_AM_DM[1], DM = SEG_AM_DM[2],gnn_model = GNN_model).to(device)
    print_model_parm_nums(model)

    basic_path = save_result_path
    if not os.path.exists(basic_path):
        os.mkdir(basic_path)

    model.load_state_dict(torch.load(model_load_path))
    test_results = test_net_dice_w_2dproj(model_load_path, basic_path, model, switch, modelDataLoader['test'],
                                         num_class, shape, TSNE, gpu=True)

    return val_dice, test_results

def test_net_dice_w_2dproj(model_load_path, basic_path, model, network_switch, dataset,num_class,shape, TSNE, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        tSNE = []
        tSNE_labels = []

        for i, b in enumerate(tqdm(dataset)):
            case_name = b['case_name'][0].split('/')[-1]
            print('case: ',case_name)
            image = sitk.ReadImage(b['case_name'][0])
            image_array = sitk.GetArrayFromImage(image)

            img = b['image'].float().squeeze()

            prediction = torch.zeros((16,128,160,160), dtype=torch.float)

            padding_z = img.shape[0] - image_array.shape[0]
            padding_y = img.shape[1] - image_array.shape[1]
            padding_x = img.shape[2] - image_array.shape[2]
            # print(image_array.shape,img.shape,padding_z, padding_y, padding_x)
            img = img[np.newaxis,np.newaxis,  :, :, :]
            imgShape = img.shape[-3:]

            resultShape = shape
            print(resultShape)
            # print(np.array(imgShape),np.array(resultShape),(np.array(imgShape) < np.array(resultShape)).any())

            if (np.array(imgShape) > np.array(resultShape)).any():
                continue
                overlapZ = 0.5
                overlapH = 0.5
                overlapW = 0.5

                interZ = int(resultShape[0] * (1.0 - overlapZ))
                interH = int(resultShape[1] * (1.0 - overlapH))
                interW = int(resultShape[2] * (1.0 - overlapW))

                iterZ = int(((imgShape[0]-resultShape[0]) / interZ)+1)
                iterH = int(((imgShape[1]-resultShape[1]) / interH)+1)
                iterW = int(((imgShape[2]-resultShape[2]) / interW)+1)

                freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
                freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
                freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

                startZ = int(freeZ/2)
                startH = int(freeH/2)
                startW = int(freeW/2)

                if num_class == 1:
                    imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
                    resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

                    for z in range(0, iterZ):
                        for h in range(0, iterH):
                            for w in range(0, iterW):
                                input = img[:, :, (startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                        (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                        (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                                if gpu:
                                    input = input.cuda()

                                if TSNE:
                                    outputsL = model(input, phase, network_switch)[2][0].cpu()

                                    layers = 1

                                    for i in range(1, layers):
                                        mask_label = skimage.measure.block_reduce(mask_label, (2, 2, 2), np.mean)

                                    for i in range(outputsL.shape[1]):
                                        for j in range(outputsL.shape[2]):
                                            for k in range(outputsL.shape[3]):

                                                sample = outputsL.detach().numpy()[:, i, j, k].flatten()
                                                tSNE.append(sample)

                                                sample_label = mask_label[i, j, k]
                                                tSNE_labels.append(sample_label)

                                else:
                                    outputsL = model(input, phase=phase, network_switch=network_switch)[0]

                                if not TSNE:
                                    prediction[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                            (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                            (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))] += outputsL[0][0].cpu()
                                    imgMatrix[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                            (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                            (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))] += resultMatrix
                else:
                    imgMatrix = np.zeros([num_class,imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
                    resultMatrix = np.ones([num_class,resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)
                    for z in range(0, iterZ):
                        for h in range(0, iterH):
                            for w in range(0, iterW):
                                input = img[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                                        (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                                        (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))]

                                if gpu:
                                    input = input.cuda()

                                if TSNE:
                                    outputsL = model(input, phase, network_switch)[2][0].cpu()
                                    # for i in range(mask_pred.shape[0]):
                                    #     for j in range(mask_pred.shape[2]):
                                    #         for k in range(mask_pred.shape[3]):
                                    #     sample = mask_pred.detach().numpy()[i].flatten()

                                    layers = 1

                                    for i in range(1, layers):
                                        mask_label = skimage.measure.block_reduce(mask_label, (2, 2, 2), np.mean)

                                    for i in range(outputsL.shape[1]):
                                        for j in range(outputsL.shape[2]):
                                            for k in range(outputsL.shape[3]):
                                                sample = outputsL.detach().numpy()[:, i, j, k].flatten()
                                                tSNE.append(sample)

                                                sample_label = mask_label[i, j, k]
                                                tSNE_labels.append(sample_label)

                                else:
                                    outputsL = model(input, phase=phase, network_switch=network_switch)[0]

                                if not TSNE:
                                    for i in range(num_class):
                                        print(prediction.shape, imgMatrix.shape)
                                        prediction[i][(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                                        (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                                        (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += outputsL[0][i].cpu()
                                        imgMatrix[i][(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                                        (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                                        (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += resultMatrix[i]


                if not TSNE:
                    imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
                    result = np.divide(prediction.cpu().detach().numpy(), imgMatrix)
                    # print(prediction.cpu().detach().numpy().shape, imgMatrix.shape)
                    if num_class == 1:
                        result_image = np.where(result > 0.5, 1, 0).astype(int)
                    else:
                        result_image = np.argmax(result, axis=0).astype(np.uint8)
            else:
                # img = torch.from_numpy(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                with autocast(enabled=True):
                    with torch.no_grad():
                        predict= model(img, phase=phase, network_switch=network_switch)[0].data.cpu().numpy().squeeze()

                if num_class == 1:
                    result_image = np.where(predict > 0.5, 1, 0).astype(int)
                else:
                    result_image = np.argmax(predict, axis=0).astype(np.uint8)


                result_image = result_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]



            result = sitk.GetImageFromArray(result_image.astype(np.uint8))
            result.SetOrigin(image.GetOrigin())
            result.SetSpacing(image.GetSpacing())
            result.SetDirection(image.GetDirection())

            sitk.WriteImage(result,os.path.join(basic_path,case_name))
            print('3d Segmentation',np.unique(result_image))


def quadrant_tooth_parsing(num_classes,shape,test_dir,checkpoint_dir_ToothSeg,save_result_path):
    model_load_path = checkpoint_dir_ToothSeg
    device, data_sizes, modelDataloader = Tooth_dataloader_Project_GCN_Child.Toothdata(test_dir,shape,num_classes)
    network_training_epoch(model_load_path,save_result_path,device, data_sizes,modelDataloader, num_classes, shape,
                                                                                             False,'GCN',[1,1,0])




