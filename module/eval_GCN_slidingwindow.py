import os.path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.cuda.amp import autocast,GradScaler
from module.dice_loss import dice_coeff
from module.visualize_attention import visualize_Seg, visualize_Rec
import module.common_module as cm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import module.evaluation_voxel as evaluation
from module.DAML import AM_Generate_Pred
# import module.evaluation_lesion as evaluation
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import skimage.measure
from tqdm import trange
from copy import deepcopy
import time

def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

def metric_calculate(pred_mask, label, num_classes = 10):
    tp, tn, fp, fn = 0, 0, 0, 0
    if num_classes != 1:
        for i in range(1,num_classes):
            tp += torch.sum(label[i] * pred_mask[i])
            tn += torch.sum((1 - label[i]) * (1 - pred_mask[i]))
            fp += torch.sum((1 - label[i]) * pred_mask[i])
            fn += torch.sum(label[i] * (1 - pred_mask[i]))
    else:
        tp += torch.sum(label * pred_mask)
        tn += torch.sum((1 - label) * (1 - pred_mask))
        fp += torch.sum((1 - label) * pred_mask)
        fn += torch.sum(label * (1 - pred_mask))

    dice_tptn = (2 * tp + 10e-4) / (2 * tp + fp + 1 * fn + 10e-4)
    acc = (tp + tn + 10e-4) / (tp + tn + fp + fn + 10e-4)
    sen = (tp + 10e-4) / (tp + fn + 10e-4)
    pre = (tp + 10e-4) / (tp + fp + 10e-4)
    # print(dice1, dice_tptn, sen, pre)
    return float(dice_tptn),float(sen),float(pre)

def eval_net_hm(net, net_copy, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None,
                  epoch=None, slice=20, root_path='no_root_path',DM_VAL_Epoch = None, model_iterative_path = None):
    """Evaluation without the densecrf with the dice coefficient"""
    with torch.no_grad():
        net.eval()
        tot = 0
        running_loss = 0
        for i, b in enumerate(dataset):
            img = b['image'].float()
            true_mask = b['distance_map'].float()
            case_name = b['case_name']
            if gpu:
                img = img.cuda()
                true_mask = true_mask.cuda()

            if epoch >= DM_VAL_Epoch[0]:
                img = iterative_net_input(net_copy, phase, network_switch, img, case_name, model_iterative_path)

            mask_pred = net(img, phase, network_switch)[-1]

            loss = criterion[3](mask_pred.squeeze().float(), true_mask.squeeze().float())
            mask_pred = (mask_pred > 0.2).float()
            true_mask = (true_mask > 0.2).float()

            tot += dice_coeff(mask_pred, true_mask).item()


            running_loss += loss.item() * img.size(0)

        if i == 0:
            tot = tot
            running_loss = running_loss
        else:
            tot = tot / (i + 1)
            running_loss = running_loss / (i + 1)
    return tot, running_loss

def eval_net_AM(net, net_copy, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None,
                  epoch=None, slice=20, root_path='no_root_path',AM_decoder = 'FC',DM_VAL_Epoch = None, model_iterative_path = None):
    AM_Memory_Bank = dict()
    with torch.no_grad():
        net.eval()
        tot = 0
        running_loss = 0
        count = 0
        correct_count = 0
        for i, b in enumerate(dataset):
            img = b['image'].float()
            AM = b['am']
            case_name = b['case_name']


            if gpu:
                img = img.cuda()
                AM = AM.cuda()

            if epoch >= DM_VAL_Epoch[0]:
                img = iterative_net_input(net_copy, phase, network_switch, img, case_name, model_iterative_path)

            predict = F.softmax(net(img, phase, network_switch)[-2],1)
            count += 1
            AM_pred = AM_Generate_Pred(predict)

            if (AM == AM_pred).all():
                correct_count += 1

            case_name = tuple(case_name)
            AM_Memory_Bank[case_name] = AM_pred

        if i == 0:
            tot = correct_count/count
            running_loss = running_loss
        else:
            tot = correct_count/count
            running_loss = running_loss / (i + 1)
    return tot, running_loss, AM_Memory_Bank

def eval_net_dice(net, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None, epoch=None, slice=20, root_path='no_root_path'):
    with torch.no_grad():
        net.eval()
        tot,sen,pre = 0, 0, 0
        running_loss = 0
        for i, b in enumerate(dataset):
            img = b['image'].float()
            true_mask = b['mask'].float()
            if gpu:
                img = img.cuda()
                true_mask = true_mask.cuda()
            # mask_pred,_ = net(img)
            mask_pred = net(img, phase, network_switch)[0]
            # mask_pred = (mask_pred > 0.5).float()
            mask_pred = F.softmax(mask_pred,1)
            # tot += dice_coeff(mask_pred, true_mask).item()
            # print(len(mask_pred[0]))
            tot1,sen1,pre1 = metric_calculate(mask_pred.squeeze(), true_mask.squeeze(),num_classes=len(mask_pred[0]))
            tot += tot1
            sen += sen1
            pre += pre1
            # test_image = np.transpose((true_mask.cpu().detach().numpy()[0]).astype(float), [1, 2, 0])
            # result_image = np.transpose((mask_pred.cpu().detach().numpy()[0][0]).astype(float), [1, 2, 0])
            #
            # tot += evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))[0]

            loss = criterion[0](mask_pred.float(), true_mask.float())
            running_loss += loss.item() * img.size(0)

            if preview:
                if visualize_batch is not None:
                    if i == int(visualize_batch):
                        outputs_vis = mask_pred.cpu().detach().numpy()
                        inputs_vis = img.cpu().detach().numpy()
                        labels_vis = true_mask.cpu().detach().numpy()
                        fig = visualize_Seg(inputs_vis[0][0], labels_vis[0], outputs_vis[0][0], figsize=(6, 6), epoch=epoch, slice=slice)

                        cm.mkdir(root_path + 'preview/val/Labeled')
                        plt.savefig(root_path + 'preview/val/Labeled/' + 'epoch_%s.jpg' % epoch)
                        # plt.show(block=False)
                        # plt.pause(1.0)
                        plt.close(fig)

        if i == 0:
            tot = tot
            running_loss = running_loss
        else:
            tot = tot / (i+1)
            sen = sen / (i+1)
            pre = pre / (i+1)
            running_loss = running_loss / (i+1)
            # print(tot,sen,pre)
    return tot, sen,pre,running_loss

def eval_net_dice_am(net, net_copy, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None, epoch=None, slice=20, root_path='no_root_path',DM_VAL_Epoch = None, model_iterative_path = None):
    with torch.no_grad():
        net.eval()
        tot,sen,pre = 0, 0, 0
        running_loss = 0
        sum_count = 0
        count = 0
        for i, b in enumerate(dataset):
            img = b['image'].float()
            true_mask = b['mask'].float()
            case_name = b['case_name']
            AM = b['am']
            case_name = tuple(case_name)

            if gpu:
                img = img.cuda()
                true_mask = true_mask.cuda()
                AM = AM.cuda()

            if epoch >= DM_VAL_Epoch[0]:
                img = iterative_net_input(net_copy, phase, network_switch, img, case_name, model_iterative_path)

            mask_pred = F.softmax(net(img, phase, network_switch)[0],1)
            mmip_pred = F.softmax(net(img, phase, network_switch)[-2],1)

            AM_pred = AM_Generate_Pred(mmip_pred)
            if (AM.squeeze() == AM_pred.squeeze()).all():
                count += 1

            sum_count += 1

            # mask_pred = (mask_pred > 0.5).float()

            # tot += dice_coeff(mask_pred, true_mask).item()
            # print(len(mask_pred[0]))
            tot1, sen1, pre1 = metric_calculate(mask_pred.squeeze(), true_mask.squeeze(), num_classes=len(mask_pred[0]))
            tot += tot1
            sen += sen1
            pre += pre1
            # test_image = np.transpose((true_mask.cpu().detach().numpy()[0]).astype(float), [1, 2, 0])
            # result_image = np.transpose((mask_pred.cpu().detach().numpy()[0][0]).astype(float), [1, 2, 0])
            #
            # tot += evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))[0]

            loss = criterion[0](mask_pred.float(), true_mask.float())
            running_loss += loss.item() * img.size(0)


        if i == 0:
            tot = tot
            running_loss = running_loss
        else:
            tot = tot / (i+1)
            sen = sen / (i+1)
            pre = pre / (i+1)
            acc = count / sum_count
            running_loss = running_loss / (i+1)
            # print(tot,sen,pre)
    return tot, sen,pre,acc,running_loss


def eval_net_mse(net, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None, epoch=None, slice=20, root_path='no_root_path'):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    with torch.no_grad():
        running_loss = 0
        for i, b in enumerate(dataset):
            img = b['image'].float()
            true_mask = b['image'].float()

            if gpu:
                img = img.cuda()
                true_mask = true_mask.cuda()

            mask_predL, _, mask_pred = net(img, phase, network_switch)
            # mask_pred = (mask_pred > 0.5).float()
            mask_pred = mask_pred.float()

            outputsU_back = mask_pred[:, 0:1]
            outputsU_1 = mask_pred[:, 1:2]
            labels_back = (1.0 - mask_predL) * true_mask.float()
            labels_1 = mask_predL * true_mask.float()

            loss = criterion[1](outputsU_1.float(), labels_1.float())
            running_loss += loss.item() * img.size(0)

            if preview:
                if visualize_batch is not None:
                    if i == int(visualize_batch):
                        outputsU_back_vis = outputsU_back.cpu().detach().numpy()
                        outputsU_1_vis = outputsU_1.cpu().detach().numpy()
                        inputs_vis = img.cpu().detach().numpy()
                        labels_back_vis = labels_back.cpu().detach().numpy()
                        labels_1_vis = labels_1.cpu().detach().numpy()
                        fig = visualize_Rec(inputs_vis[0][0], labels_back_vis[0, 0], labels_1_vis[0, 0],
                                            outputsU_back_vis[0, 0], outputsU_1_vis[0, 0], figsize=(6, 6), epoch=epoch)
                        cm.mkdir(root_path + 'preview/val/Unlabeled')
                        plt.savefig(root_path + 'preview/val/Unlabeled/' + 'epoch_%s.jpg' % epoch)
                        # plt.show(block=False)
                        # plt.pause(1.0)
                        plt.close(fig)

        if i == 0:
            running_loss = running_loss
        else:
            running_loss = running_loss / (i+1)

    return running_loss, running_loss


def test_net_dice_w_2dproj(model_load_path, save_path,basic_path, model, network_switch, dataset,num_class,shape, TSNE, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        DSC, AVD, Recall, F1 = 0, 0, 0, 0
        iter = 0
        i = 0

        tSNE = []
        tSNE_labels = []
        count,correct_count = 0,0
        file = open(basic_path + '/test_results_lists.txt', 'a')
        AM_MSE = 0
        for i, b in enumerate(tqdm(dataset)):
            start_time = time.time()
            case_name = b['case_name'][0].split('/')[-1]
            print('case: ',case_name)
            image = sitk.ReadImage(b['case_name'][0])
            image_array = sitk.GetArrayFromImage(image)

            DSC_1, AVD_1, Recall_1, F1_1 = 0, 0, 0, 0
            img = b['image'].float().squeeze()
            true_mask = b['mask'][0]
            AM = b['am']
            mip_label = b['mmip_mask']
            # create prediction tensor
            prediction = torch.zeros(true_mask.size(), dtype=torch.float)
            # print(img.shape)
            # padding_z = (16 - img.shape[0] % 16) % 16
            # img = np.pad(img, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
            # padding_y = (16 - img.shape[1] % 16) % 16
            # img = np.pad(img, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
            # padding_x = (16 - img.shape[2] % 16) % 16
            # img = np.pad(img, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
            # print(shape, img.shape)


            # padding_z = shape[0] - img.shape[0]
            # img = np.pad(img, ((0, padding_z), (0, 0), (0, 0)), 'constant')
            # padding_y = shape[1] - img.shape[1]
            # img = np.pad(img, ((0, 0), (0, padding_y), (0, 0)), 'constant')
            # padding_x = shape[2] - img.shape[2]
            # img = np.pad(img, ((0, 0), (0, 0), (0, padding_x)), 'constant')
            # print(padding_z,padding_y,padding_x)
            padding_z = img.shape[0] - image_array.shape[0]
            padding_y = img.shape[1] - image_array.shape[1]
            padding_x = img.shape[2] - image_array.shape[2]
            # print(image_array.shape,img.shape,padding_z, padding_y, padding_x)
            img = img[np.newaxis,np.newaxis,  :, :, :]
            imgShape = img.shape[-3:]

            resultShape = shape
            # print(np.array(imgShape),np.array(resultShape),(np.array(imgShape) < np.array(resultShape)).any())

            if (np.array(imgShape) > np.array(resultShape)).any():
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

                                label = true_mask[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                        (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                        (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                                if gpu:
                                    input = input.cuda()

                                if TSNE:
                                    outputsL = model(input, phase, network_switch)[2][0].cpu()
                                    # for i in range(mask_pred.shape[0]):
                                    #     for j in range(mask_pred.shape[2]):
                                    #         for k in range(mask_pred.shape[3]):
                                    #     sample = mask_pred.detach().numpy()[i].flatten()
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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

                                label = true_mask[(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
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
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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
                        test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                        result_image = np.where(result > 0.5, 1, 0).astype(int)
                    else:
                        result_image = np.argmax(result, axis=0).astype(np.uint8)
                        test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)
            else:
                # img = torch.from_numpy(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                with autocast(enabled=True):
                    with torch.no_grad():
                        predict= model(img, phase=phase, network_switch=network_switch)[0].data.cpu().numpy().squeeze()
                        fea_logit_3 = F.softmax(model(img, phase=phase, network_switch=network_switch)[1].data.cpu().float(),1)
                        fea_logit_3 = F.interpolate(fea_logit_3, size=true_mask.size()[1:], mode='trilinear',align_corners=True).numpy().squeeze()

                        mmip_prediction =  F.softmax(model(img, phase=phase, network_switch=network_switch)[-2].data.cpu().float(),1)
                        out = mmip_prediction.data.cpu().numpy().squeeze()
                        AM_pred = AM_Generate_Pred(mmip_prediction).squeeze().numpy()
                        AM = AM.squeeze().numpy()
                        print('*' * 10, 'Predict', '*' * 10)
                        print(AM_pred)
                        print('*' * 10, 'GT', '*' * 10)
                        print(AM)

                        if check_symmetric(AM_pred):
                            MSE = np.sum(pow((AM_pred[:-1,:-1] - AM[:-1,:-1]), 2))/2
                        else:
                            MSE = np.sum(pow((AM_pred[:-1,:-1] - AM[:-1,:-1]), 2))

                        AM_MSE += MSE
                        if (AM == AM_pred).all():
                            print('True')
                            correct_count += 1
                        count += 1
                        # fea_logit_2 = F.softmax(model(img, phase=phase, network_switch=network_switch)[1][1].data.cpu())
                        # fea_logit_2 = F.interpolate(fea_logit_2, size=true_mask.size()[1:], mode='trilinear',align_corners=True).numpy().squeeze()
                        #
                        # fea_logit_1 = F.softmax(model(img, phase=phase, network_switch=network_switch)[1][2].data.cpu())
                        # fea_logit_1 = F.interpolate(fea_logit_1, size=true_mask.size()[1:], mode='trilinear',align_corners=True).numpy().squeeze()
                if num_class == 1:
                    result_image = np.where(predict > 0.5, 1, 0).astype(int)
                    test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                else:
                    result_image = np.argmax(predict, axis=0).astype(np.uint8)
                    out1_1 = np.argmax(out, axis=0).astype(np.uint8)

                    mip_label = mip_label.data.cpu().numpy().squeeze()
                    mip_label_1 =  np.argmax(mip_label, axis=0).astype(np.uint8)
                    # print(predict.shape,result_image.shape)
                    test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)



                # result_image = result_image[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # test_image = test_image[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # predict = predict[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # feature_map_3 = fea_logit_3[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # feature_map_2 = fea_logit_2[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # feature_map_1 = fea_logit_1[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # print(result_image.shape, test_image.shape)
                # print(img.shape)

                result_image = result_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                test_image = test_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                feature_map_3 = fea_logit_3[:,padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                # print(result_image.shape,test_image.shape,feature_map_3.shape)


            if not os.path.exists(os.path.join(basic_path,'result','feature_logit')):
                os.mkdir(os.path.join(basic_path,'result','feature_logit'))
            if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path)):
                os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path))
            if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder3')):
                os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder3'))
            # if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder2')):
            #     os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder2'))
            # if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder1')):
            #     os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder1'))

            # feature_map_set = [feature_map_3,feature_map_2,feature_map_1]
            # feature_decoder_name = ['decoder3','decoder2','decoder1']
            feature_map_set = [feature_map_3]
            feature_decoder_name = ['decoder3']
            for fea_idx,feature_map in enumerate(feature_map_set):
                if num_class > 13:
                    imgs_1 = torch.cat([torch.Tensor(image_array[30,:,:]/255),torch.Tensor(feature_map[0][30,:,:]),torch.Tensor(feature_map[1][30,:,:]),torch.Tensor(feature_map[2][30,:,:]),torch.Tensor(feature_map[3][30,:,:])], dim=-1)
                    imgs_2 = torch.cat([torch.Tensor(feature_map[4][30,:,:]),torch.Tensor(feature_map[5][30,:,:]),torch.Tensor(feature_map[6][30,:,:]),torch.Tensor(feature_map[7][30,:,:]),torch.Tensor(feature_map[8][30,:,:])], dim=-1)
                    imgs_3 = torch.cat([torch.Tensor(feature_map[9][30,:,:]),torch.Tensor(feature_map[10][30,:,:]),torch.Tensor(feature_map[11][30,:,:]),torch.Tensor(feature_map[12][30,:,:]),torch.Tensor(feature_map[13][30,:,:])], dim=-1)
                    imgs_concat = torch.cat((imgs_1, imgs_2, imgs_3), dim=0)
                else:
                    imgs_1 = torch.cat([torch.Tensor(image_array[30,:,:]/255),torch.Tensor(feature_map[0][30,:,:]),torch.Tensor(feature_map[1][30,:,:]),torch.Tensor(feature_map[2][30,:,:]),torch.Tensor(feature_map[3][30,:,:])], dim=-1)
                    imgs_2 = torch.cat([torch.Tensor(feature_map[4][30,:,:]),torch.Tensor(feature_map[5][30,:,:]),torch.Tensor(feature_map[6][30,:,:]),torch.Tensor(feature_map[7][30,:,:]),torch.Tensor(feature_map[8][30,:,:])], dim=-1)
                    imgs_concat = torch.cat((imgs_1, imgs_2), dim=0)
                save_image(imgs_concat, os.path.join(basic_path, 'result', 'feature_logit',save_path,feature_decoder_name[fea_idx],case_name.replace('.nii.gz', '.png')), nrow=1)

            if num_class > 13:
                imgs_1 = torch.cat([torch.Tensor(out1_1), torch.Tensor(out[0]), torch.Tensor(out[1]), torch.Tensor(out[2]),
                                    torch.Tensor(out[3])], dim=-1)

                imgs_2 = torch.cat([torch.Tensor(mip_label_1), torch.Tensor(mip_label[0]), torch.Tensor(mip_label[1]),
                                    torch.Tensor(mip_label[2]),
                                    torch.Tensor(mip_label[3])], dim=-1)
                imgs_3 = torch.cat([torch.Tensor(out[4]), torch.Tensor(out[5]), torch.Tensor(out[6]),
                                    torch.Tensor(out[7]), torch.Tensor(out[8])], dim=-1)

                imgs_4 = torch.cat([torch.Tensor(mip_label[4]), torch.Tensor(mip_label[5]),
                                    torch.Tensor(mip_label[6]),
                                    torch.Tensor(mip_label[7]), torch.Tensor(mip_label[8])], dim=-1)
                imgs_5 = torch.cat([torch.Tensor(out[9]), torch.Tensor(out[10]), torch.Tensor(out[11]),
                                    torch.Tensor(out[11]), torch.Tensor(out[11])], dim=-1)

                imgs_6 = torch.cat([torch.Tensor(mip_label[9]), torch.Tensor(mip_label[10]),
                                    torch.Tensor(mip_label[11]),
                                    torch.Tensor(mip_label[12]), torch.Tensor(mip_label[13])], dim=-1)
                imgs_concat = torch.cat((imgs_1, imgs_2, imgs_3, imgs_4, imgs_5, imgs_6), dim=0)
            else:
                imgs_1 = torch.cat(
                    [torch.Tensor(out1_1), torch.Tensor(out[0]), torch.Tensor(out[1]), torch.Tensor(out[2]),
                     torch.Tensor(out[3])], dim=-1)

                imgs_2 = torch.cat([torch.Tensor(mip_label_1), torch.Tensor(mip_label[0]), torch.Tensor(mip_label[1]),
                                    torch.Tensor(mip_label[2]),
                                    torch.Tensor(mip_label[3])], dim=-1)
                imgs_3 = torch.cat([torch.Tensor(out[4]), torch.Tensor(out[5]), torch.Tensor(out[6]),
                                    torch.Tensor(out[7]), torch.Tensor(out[8])], dim=-1)

                imgs_4 = torch.cat([torch.Tensor(mip_label[4]), torch.Tensor(mip_label[5]),
                                    torch.Tensor(mip_label[6]),
                                    torch.Tensor(mip_label[7]), torch.Tensor(mip_label[8])], dim=-1)
                imgs_concat = torch.cat((imgs_1, imgs_2, imgs_3, imgs_4), dim=0)

            dsc, avd, recall, f1 = evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))
            DSC += dsc
            AVD += avd
            Recall += recall
            F1 += f1

            DSC_1 += dsc
            AVD_1 += avd
            Recall_1 += recall
            F1_1 += f1

            history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(DSC_1, AVD_1, Recall_1, F1_1))
            file.write(history)

            result = sitk.GetImageFromArray(result_image.astype(np.uint8))
            result.SetOrigin(image.GetOrigin())
            result.SetSpacing(image.GetSpacing())
            result.SetDirection(image.GetDirection())
            if not os.path.exists(os.path.join(basic_path,'result',save_path)):
                os.mkdir(os.path.join(basic_path,'result',save_path))
            AM_type = 'proposed'
            if not os.path.exists(os.path.join(basic_path,'result',save_path,AM_type)):
                os.mkdir(os.path.join(basic_path,'result',save_path,AM_type))
            seg_path = os.path.join(basic_path,'result',save_path,AM_type)
            if not os.path.exists(os.path.join(basic_path, 'result', save_path, '2d_projection')):
                os.mkdir(os.path.join(basic_path, 'result', save_path, '2d_projection'))
            pro_path = os.path.join(basic_path, 'result', save_path, '2d_projection')
            print(os.path.join(seg_path,case_name))
            sitk.WriteImage(result,os.path.join(seg_path,case_name))
            save_image(imgs_concat,os.path.join(pro_path,case_name.replace('.nii.gz', '.png')),nrow=1)
            print('AM MSE',MSE)
            print('3d Segmentation',np.unique(result_image))
            print('3d Ground Truth', np.unique(test_image))
            print('2d Projection', np.unique(out1_1))
            end_time = time.time()
            print('Per case time:', end_time - start_time)
            # feature_dir = r'/media/ps/lys/CBCT_tooth/Project_GCN/Quadrant/train/feature_map'
            # np.save(os.path.join(feature_dir,case_name.replace('.nii.gz','.npy')),feature_map)
            # print(feature_map.shape)
        file.close()
        print('Correct/All: {}/{}'.format(correct_count, count), ' Acc: ', round(correct_count / count, 4), 'AM MSE: ',round(AM_MSE / count, 4))

        total = (iter + 1) * (i + 1)

    return DSC/total, AVD/total, Recall/total, F1/total

def test_net_dice(model_load_path, save_path,basic_path, model, network_switch, dataset,num_class,shape, TSNE, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        DSC, AVD, Recall, F1 = 0, 0, 0, 0
        iter = 0
        i = 0

        tSNE = []
        tSNE_labels = []
        count,correct_count = 0,0
        file = open(basic_path + '/test_results_lists.txt', 'a')

        for i, b in enumerate(tqdm(dataset)):
            case_name = b['case_name'][0].split('/')[-1]
            print('case: ',case_name)
            image = sitk.ReadImage(b['case_name'][0])
            image_array = sitk.GetArrayFromImage(image)

            DSC_1, AVD_1, Recall_1, F1_1 = 0, 0, 0, 0
            img = b['image'].float().squeeze()
            true_mask = b['mask'][0]
            AM = b['am']
            mip_label = b['mmip_mask']
            # create prediction tensor
            prediction = torch.zeros(true_mask.size(), dtype=torch.float)
            # print(img.shape)
            # padding_z = (16 - img.shape[0] % 16) % 16
            # img = np.pad(img, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
            # padding_y = (16 - img.shape[1] % 16) % 16
            # img = np.pad(img, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
            # padding_x = (16 - img.shape[2] % 16) % 16
            # img = np.pad(img, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
            # print(shape, img.shape)


            # padding_z = shape[0] - img.shape[0]
            # img = np.pad(img, ((0, padding_z), (0, 0), (0, 0)), 'constant')
            # padding_y = shape[1] - img.shape[1]
            # img = np.pad(img, ((0, 0), (0, padding_y), (0, 0)), 'constant')
            # padding_x = shape[2] - img.shape[2]
            # img = np.pad(img, ((0, 0), (0, 0), (0, padding_x)), 'constant')
            # print(padding_z,padding_y,padding_x)
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

                                label = true_mask[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                        (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                        (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                                if gpu:
                                    input = input.cuda()

                                if TSNE:
                                    outputsL = model(input, phase, network_switch)[2][0].cpu()
                                    # for i in range(mask_pred.shape[0]):
                                    #     for j in range(mask_pred.shape[2]):
                                    #         for k in range(mask_pred.shape[3]):
                                    #     sample = mask_pred.detach().numpy()[i].flatten()
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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

                                label = true_mask[(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
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
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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
                        test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                        result_image = np.where(result > 0.5, 1, 0).astype(int)
                    else:
                        result_image = np.argmax(result, axis=0).astype(np.uint8)
                        test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)
            else:
                # img = torch.from_numpy(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                with autocast(enabled=True):
                    with torch.no_grad():
                        predict= model(img, phase=phase, network_switch=network_switch)[0].data.cpu().numpy().squeeze()
                        fea_logit_3 = F.softmax(model(img, phase=phase, network_switch=network_switch)[1].data.cpu().float(),1)
                        fea_logit_3 = F.interpolate(fea_logit_3, size=true_mask.size()[1:], mode='trilinear',align_corners=True).numpy().squeeze()

                        # fea_logit_2 = F.softmax(model(img, phase=phase, network_switch=network_switch)[1][1].data.cpu())
                        # fea_logit_2 = F.interpolate(fea_logit_2, size=true_mask.size()[1:], mode='trilinear',align_corners=True).numpy().squeeze()
                        #
                        # fea_logit_1 = F.softmax(model(img, phase=phase, network_switch=network_switch)[1][2].data.cpu())
                        # fea_logit_1 = F.interpolate(fea_logit_1, size=true_mask.size()[1:], mode='trilinear',align_corners=True).numpy().squeeze()
                if num_class == 1:
                    result_image = np.where(predict > 0.5, 1, 0).astype(int)
                    test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                else:
                    result_image = np.argmax(predict, axis=0).astype(np.uint8)
                    mip_label = mip_label.data.cpu().numpy().squeeze()
                    # print(predict.shape,result_image.shape)
                    test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)



                # result_image = result_image[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # test_image = test_image[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # predict = predict[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # feature_map_3 = fea_logit_3[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # feature_map_2 = fea_logit_2[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # feature_map_1 = fea_logit_1[:,0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                # print(result_image.shape, test_image.shape)
                # print(img.shape)

                result_image = result_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                test_image = test_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                feature_map_3 = fea_logit_3[:,padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                            padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                            padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                # print(result_image.shape,test_image.shape,feature_map_3.shape)
            if not os.path.exists(os.path.join(basic_path,'result','feature_logit')):
                os.mkdir(os.path.join(basic_path,'result','feature_logit'))
            if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path)):
                os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path))
            if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder3')):
                os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder3'))
            # if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder2')):
            #     os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder2'))
            # if not os.path.exists(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder1')):
            #     os.mkdir(os.path.join(basic_path, 'result', 'feature_logit',save_path,'decoder1'))

            # feature_map_set = [feature_map_3,feature_map_2,feature_map_1]
            # feature_decoder_name = ['decoder3','decoder2','decoder1']
            feature_map_set = [feature_map_3]
            feature_decoder_name = ['decoder3']
            for fea_idx,feature_map in enumerate(feature_map_set):
                imgs_1 = torch.cat([torch.Tensor(image_array[30,:,:]/255),torch.Tensor(feature_map[0][30,:,:]),torch.Tensor(feature_map[1][30,:,:]),torch.Tensor(feature_map[2][30,:,:]),torch.Tensor(feature_map[3][30,:,:])], dim=-1)
                imgs_2 = torch.cat([torch.Tensor(feature_map[4][30,:,:]),torch.Tensor(feature_map[5][30,:,:]),torch.Tensor(feature_map[6][30,:,:]),torch.Tensor(feature_map[7][30,:,:]),torch.Tensor(feature_map[8][30,:,:])], dim=-1)
                imgs_3 = torch.cat([torch.Tensor(feature_map[9][30,:,:]),torch.Tensor(feature_map[10][30,:,:]),torch.Tensor(feature_map[11][30,:,:]),torch.Tensor(feature_map[12][30,:,:]),torch.Tensor(feature_map[13][30,:,:])], dim=-1)
                imgs_concat = torch.cat((imgs_1, imgs_2, imgs_3), dim=0)
                save_image(imgs_concat, os.path.join(basic_path, 'result', 'feature_logit',save_path,feature_decoder_name[fea_idx],case_name.replace('.nii.gz', '.png')), nrow=1)

            dsc, avd, recall, f1 = evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))
            DSC += dsc
            AVD += avd
            Recall += recall
            F1 += f1

            DSC_1 += dsc
            AVD_1 += avd
            Recall_1 += recall
            F1_1 += f1

            history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(DSC_1, AVD_1, Recall_1, F1_1))
            file.write(history)

            result = sitk.GetImageFromArray(result_image.astype(np.uint8))
            result.SetOrigin(image.GetOrigin())
            result.SetSpacing(image.GetSpacing())
            result.SetDirection(image.GetDirection())
            if not os.path.exists(os.path.join(basic_path,'result',save_path)):
                os.mkdir(os.path.join(basic_path,'result',save_path))
            if not os.path.exists(os.path.join(basic_path,'result',save_path,'proposed')):
                os.mkdir(os.path.join(basic_path,'result',save_path,'proposed'))
            seg_path = os.path.join(basic_path,'result',save_path,'proposed')
            print(os.path.join(seg_path,case_name))
            sitk.WriteImage(result,os.path.join(seg_path,case_name))

            print('3d Segmentation',np.unique(result_image))
            print('3d Ground Truth', np.unique(test_image))
            # feature_dir = r'/media/ps/lys/CBCT_tooth/Project_GCN/Quadrant/train/feature_map'
            # np.save(os.path.join(feature_dir,case_name.replace('.nii.gz','.npy')),feature_map)
            # print(feature_map.shape)
        file.close()

        total = (iter + 1) * (i + 1)

    return DSC/total, AVD/total, Recall/total, F1/total

Adjacency_Matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def test_net_am_acc(model_load_path, save_path,basic_path, model, model_copy, network_switch, dataset,num_class,shape, TSNE,model_iterative_path,  gpu=False,AM_decoder = 'FC'):
    AM_Memory_Bank = dict()
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        correct_count = 0
        count = 0
        DSC, AVD, Recall, F1 = 0, 0, 0, 0
        for i, b in enumerate(tqdm(dataset)):
            case_name = b['case_name'][0].split('/')[-1]
            case_name_key = b['case_name']
            print('case: ',case_name)
            image = sitk.ReadImage(b['case_name'][0])
            image_array = sitk.GetArrayFromImage(image)

            img = b['image'].float().squeeze()
            true_mask = b['mask'][0]
            AM = b['am']
            AM = AM.squeeze().cpu().detach().numpy()
            # if (AM != Adjacency_Matrix).any():
            #     continue
            # if (AM == Adjacency_Matrix).all():
            #     continue

            # create prediction tensor
            prediction = torch.zeros(true_mask.size(), dtype=torch.float)
            if img.shape[0] != 2:
                img = img[np.newaxis, np.newaxis, :, :, :]
            else:
                img = img[np.newaxis, :, :, :]

            input = img.cuda()
            if os.path.exists(model_iterative_path):
                input = iterative_net_input(model_copy, phase, network_switch, input, case_name_key, model_iterative_path,True)
                print('Iterative Input Loading')
            else:
                print('Origin Input Loading')
            outputsL_am = model(input, phase=phase, network_switch=network_switch)[1]
            predict_am = torch.sigmoid(outputsL_am).squeeze().cpu().detach().numpy()
            predict_am = (predict_am > 0.5).astype(np.float32)

            if AM_decoder == 'FC':
                AM_pred = np.zeros_like(AM)
                start = 0
                end = 0
                for y in range(AM.shape[0]):
                    end += AM.shape[1] - y
                    AM_pred[y][y:] = predict_am[start:end]
                    start = end
                AM_pred = np.transpose(AM_pred) + AM_pred
                AM_pred[AM_pred > 0] = 1
            else:
                AM_pred = predict_am
                if not check_symmetric(AM_pred):
                    # print('not symmetric')
                    # print(AM_pred,check_symmetric(AM_pred))
                    AM_pred = AM_pred + np.transpose(AM_pred) * (np.transpose(AM_pred) > AM_pred) - AM_pred * (
                                                       np.transpose(AM_pred) > AM_pred)
                    AM_pred[AM_pred > 0] = 1
                    # print('after symmetric')
                    # print(AM_pred,check_symmetric(AM_pred))

            if (AM == AM_pred).all():
                correct_count += 1
            count += 1
            print('*'*10,'Predict','*'*10)
            print(AM_pred) #AM_pred
            print('check_symmetric',check_symmetric(AM_pred))
            print('*' * 10, 'Ground Truth', '*' * 10)
            print(AM)
            if (AM == AM_pred).all():
                print('True')
            else:
                print('False')
            case_name = tuple(case_name)
            AM_Memory_Bank[case_name] = AM_pred
            # AM_Memory_Bank[case_name] = Adjacency_Matrix
        print('Correct/All: {}/{}'.format(correct_count, count), ' Acc: ', round(correct_count / count, 4))

    return DSC, AVD, Recall, F1, AM_Memory_Bank

def test_net_dice_am(model_load_path, save_path,basic_path, AM_Memory_Bank,model, model_copy, network_switch, dataset,num_class,shape, TSNE,model_iterative_path, gpu=False):
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        DSC, AVD, Recall, F1 = 0, 0, 0, 0
        iter = 0
        i = 0

        tSNE = []
        tSNE_labels = []

        file = open(basic_path + '/test_results_lists.txt', 'a')

        for i, b in enumerate(tqdm(dataset)):
            case_name = b['case_name'][0].split('/')[-1]
            case_name_key = b['case_name']
            print('case: ',case_name)
            image = sitk.ReadImage(b['case_name'][0])
            image_array = sitk.GetArrayFromImage(image)

            DSC_1, AVD_1, Recall_1, F1_1 = 0, 0, 0, 0
            img = b['image'].float().squeeze()
            true_mask = b['mask'][0]
            # print('AM_Memory_Bank',AM_Memory_Bank[case_name])
            edge_indexs = []
            # print(AM_Memory_Bank)
            edge_index_temp = sp.coo_matrix(AM_Memory_Bank[tuple(case_name)])
            # edge_index_temp = sp.coo_matrix(Adjacency_Matrix)
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(indices)
            edge_indexs.append(edge_index)
            # print(edge_indexs)
            # create prediction tensor
            prediction = torch.zeros(true_mask.size(), dtype=torch.float)
            # print(img.shape)
            # padding_z = (16 - img.shape[0] % 16) % 16
            # img = np.pad(img, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
            # padding_y = (16 - img.shape[1] % 16) % 16
            # img = np.pad(img, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
            # padding_x = (16 - img.shape[2] % 16) % 16
            # img = np.pad(img, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
            # print(shape, img.shape)


            # padding_z = shape[0] - img.shape[0]
            # img = np.pad(img, ((0, padding_z), (0, 0), (0, 0)), 'constant')
            # padding_y = shape[1] - img.shape[1]
            # img = np.pad(img, ((0, 0), (0, padding_y), (0, 0)), 'constant')
            # padding_x = shape[2] - img.shape[2]
            # img = np.pad(img, ((0, 0), (0, 0), (0, padding_x)), 'constant')
            padding_z = img.shape[0] - image_array.shape[0]
            padding_y = img.shape[1] - image_array.shape[1]
            padding_x = img.shape[2] - image_array.shape[2]
            # print(padding_z,padding_y,padding_x)
            if img.shape[0] != 2:
                img = img[np.newaxis, np.newaxis, :, :, :]
            else:
                img = img[np.newaxis, :, :, :]
            imgShape = img.shape[-3:]

            img = img.cuda()
            if os.path.exists(model_iterative_path):
                img = iterative_net_input(model_copy, phase, network_switch, img, case_name_key, model_iterative_path,True)
                print('Iterative Input Loading')
            else:
                print('Origin Input Loading')
            resultShape = shape
            print(resultShape)
            # print(np.array(imgShape),np.array(resultShape),(np.array(imgShape) < np.array(resultShape)).any())

            if (np.array(imgShape) > np.array(resultShape)).any():
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

                                label = true_mask[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                        (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                        (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                                if gpu:
                                    input = input.cuda()

                                if TSNE:
                                    outputsL = model(input, phase, network_switch)[2][0].cpu()
                                    # for i in range(mask_pred.shape[0]):
                                    #     for j in range(mask_pred.shape[2]):
                                    #         for k in range(mask_pred.shape[3]):
                                    #     sample = mask_pred.detach().numpy()[i].flatten()
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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
                                    print(edge_indexs)
                                    outputsL = model(input, phase=phase, network_switch=network_switch, edge_indexs = edge_indexs)[0]

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

                                label = true_mask[(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
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
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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
                        test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                        result_image = np.where(result > 0.5, 1, 0).astype(int)
                    else:
                        result_image = np.argmax(result, axis=0).astype(np.uint8)
                        test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)
            else:
                # img = torch.from_numpy(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                with torch.no_grad():
                    predict= model(img, phase=phase, network_switch=network_switch,edge_indexs = edge_indexs)[0].data.cpu().numpy().squeeze()
                if num_class == 1:
                    result_image = np.where(predict > 0.5, 1, 0).astype(int)
                    test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                else:
                    result_image = np.argmax(predict, axis=0).astype(np.uint8)
                    # print(predict.shape,result_image.shape)
                    test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)


                # result_image = result_image[padding_z // 2: result_image.shape[0] - (padding_z - padding_z // 2),
                #             padding_y // 2: result_image.shape[1] - (padding_y - padding_y // 2),
                #             padding_x // 2: result_image.shape[2] - (padding_x - padding_x // 2)]
                result_image = result_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                               padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                               padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                test_image = test_image[padding_z // 2: imgShape[0] - (padding_z - padding_z // 2),
                               padding_y // 2: imgShape[1] - (padding_y - padding_y // 2),
                               padding_x // 2: imgShape[2] - (padding_x - padding_x // 2)]
                # print(result_image.shape, test_image.shape)

            print(np.unique(result_image))
            dsc, avd, recall, f1 = evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))
            DSC += dsc
            AVD += avd
            Recall += recall
            F1 += f1

            DSC_1 += dsc
            AVD_1 += avd
            Recall_1 += recall
            F1_1 += f1

            history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(DSC_1, AVD_1, Recall_1, F1_1))
            file.write(history)

            result = sitk.GetImageFromArray(result_image.astype(np.uint8))
            result.SetOrigin(image.GetOrigin())
            result.SetSpacing(image.GetSpacing())
            result.SetDirection(image.GetDirection())
            if not os.path.exists(os.path.join(basic_path,'result',save_path)):
                os.mkdir(os.path.join(basic_path,'result',save_path))
            print(os.path.join(basic_path,'result',save_path,case_name))
            sitk.WriteImage(result,os.path.join(basic_path,'result',save_path,case_name))

        file.close()

        total = (iter + 1) * (i + 1)

    return DSC/total, AVD/total, Recall/total, F1/total

def test_net_loc_dice(model_load_path, save_path,basic_path, model, model_copy, network_switch, dataset,num_class,shape, TSNE,model_iterative_path, gpu=False):
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        DSC, AVD, Recall, F1 = 0, 0, 0, 0
        iter = 0
        i = 0

        tSNE = []
        tSNE_labels = []

        file = open(basic_path + '/test_results_lists.txt', 'a')

        for i, b in enumerate(tqdm(dataset)):
            case_name_key = b['case_name']
            case_name = b['case_name'][0].split('/')[-1]
            print('case: ',case_name)
            image = sitk.ReadImage(b['case_name'][0])
            image_array = sitk.GetArrayFromImage(image)

            DSC_1, AVD_1, Recall_1, F1_1 = 0, 0, 0, 0
            img = b['image'].float().squeeze()
            true_mask = b['distance_map'][0]
            # print('AM_Memory_Bank',AM_Memory_Bank[case_name])

            # create prediction tensor
            prediction = torch.zeros(true_mask.size(), dtype=torch.float)

            if img.shape[0] != 2:
                img = img[np.newaxis,np.newaxis,  :, :, :]
            else:
                img = img[np.newaxis, :, :, :]

            img = img.cuda()
            if os.path.exists(model_iterative_path):
                img = iterative_net_input(model_copy, phase, network_switch, img, case_name_key, model_iterative_path,True)
                print('Iterative Input Loading')
            else:
                print('Origin Input Loading')
            imgShape = img.shape[-3:]

            resultShape = shape
            print(resultShape)
            # print(np.array(imgShape),np.array(resultShape),(np.array(imgShape) < np.array(resultShape)).any())

            if (np.array(imgShape) > np.array(resultShape)).any():
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

                                label = true_mask[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                        (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                        (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                                if gpu:
                                    input = input.cuda()

                                if TSNE:
                                    outputsL = model(input, phase, network_switch)[2][0].cpu()
                                    # for i in range(mask_pred.shape[0]):
                                    #     for j in range(mask_pred.shape[2]):
                                    #         for k in range(mask_pred.shape[3]):
                                    #     sample = mask_pred.detach().numpy()[i].flatten()
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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
                                    outputsL = model(input, phase=phase, network_switch=network_switch)[-1]

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

                                label = true_mask[(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
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
                                    #     tSNE.append(sample)
                                    mask_label = label.detach().numpy()

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
                                    outputsL = model(input, phase=phase, network_switch=network_switch)[-1]

                                if not TSNE:
                                    for i in range(num_class):
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
                        test_image = np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0).astype(int)
                        result_image = np.where(result > 0.5, 1, 0).astype(int)
                    else:
                        result_image = np.argmax(result, axis=0).astype(np.uint8)
                        test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)
            else:
                # img = torch.from_numpy(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                with torch.no_grad():
                    predict= model(img, phase=phase, network_switch=network_switch)[-1].data.cpu().numpy().squeeze()
                if num_class == 1:

                    result_image = np.where(predict > 0.5, 1, 0).astype(int)
                    test_image = np.where(true_mask.cpu().detach().numpy() > 0, 1, 0).astype(int)
                else:
                    result_image = np.argmax(predict, axis=0).astype(np.uint8)
                    # print(predict.shape,result_image.shape)
                    test_image = np.argmax(true_mask.cpu().detach().numpy(), axis=0).astype(np.uint8)


                # result_image = result_image[padding_z // 2: result_image.shape[0] - (padding_z - padding_z // 2),
                #             padding_y // 2: result_image.shape[1] - (padding_y - padding_y // 2),
                #             padding_x // 2: result_image.shape[2] - (padding_x - padding_x // 2)]
                result_image = result_image[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                test_image = test_image[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]
                predict = predict[0: image_array.shape[0],0: image_array.shape[1],0: image_array.shape[2]]

                # print(result_image.shape, test_image.shape)

            print(np.unique(result_image))
            dsc, avd, recall, f1 = evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))
            DSC += dsc
            AVD += avd
            Recall += recall
            F1 += f1

            DSC_1 += dsc
            AVD_1 += avd
            Recall_1 += recall
            F1_1 += f1

            history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(DSC_1, AVD_1, Recall_1, F1_1))
            file.write(history)

            result = sitk.GetImageFromArray(predict)
            result.SetOrigin(image.GetOrigin())
            result.SetSpacing(image.GetSpacing())
            result.SetDirection(image.GetDirection())
            if not os.path.exists(os.path.join(basic_path,'result',save_path)):
                os.mkdir(os.path.join(basic_path,'result',save_path))
            print(os.path.join(basic_path,'result',save_path,case_name))
            sitk.WriteImage(result,os.path.join(basic_path,'result',save_path,case_name))

        file.close()

        total = (iter + 1) * (i + 1)

    return DSC/total, AVD/total, Recall/total, F1/total


def iterative_net_input(net,  phase, network_switch, input, case_name, model_iterative_path,Test_Flag = False):
    with torch.no_grad():
        net.eval()
        img = input
        if Test_Flag:
            net.load_state_dict(torch.load(model_iterative_path))
        iterative_input_dm = net(img, phase, network_switch)[-1].data.cpu().detach()
        # inputs_iterative = torch.concat((img[:, 0, :].unsqueeze(1), iterative_input_dm), dim=1).data.cpu().detach().numpy()
        for item in range(len(case_name)):
            input[item,1] = iterative_input_dm[item].squeeze()
        torch.cuda.empty_cache()
        del iterative_input_dm
    return input
