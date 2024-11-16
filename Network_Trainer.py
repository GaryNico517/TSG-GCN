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
import module.common_module as cm
from module.dice_loss import DiceCoefficientLF,DiceLoss
from module.Huber_loss import MSE,Huber_loss
from module.loss_func import GeneralizedDiceLoss
from module.Crossentropy import crossentropy
from module.eval_GCN_slidingwindow import eval_net_dice_am, eval_net_dice, eval_net_AM, test_net_dice,test_net_dice_w_2dproj,test_net_am_acc,test_net_dice_am,eval_net_hm,test_net_loc_dice,iterative_net_input
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
from module.DAML import AM_Generate_Pred,ThreeD_Seg_to_TwoD_Proj,AM_Generate_Pred_nograd
import argparse
import SimpleITK as sitk
from copy import deepcopy

warnings.filterwarnings('ignore')

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train_model(model, model_copy,modelDataLoader, model_save_path, writer, device, root_path, network_switch, criterion, optimizer, scheduler,learning_rate,phase_now,
                num_epochs=25, loss_weighted=True, jointly=False, self=False, mode='fuse',AM_decoder = 'FC'):

    since = time.time()
    inputs = 0
    labels = 0
    AM = 0
    AM_flatten = 0
    image = 0
    image2 = 0
    outputsL = 0
    outputsL_loc = 0
    case_name = 0
    labels_back = 0
    labels_fore = 0

    loss = 0

    PREVIEW = False

    # dict = defaultdict(list)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_dice = 0.0
    best_val_sensitivity = 0.0
    best_val_precision = 0.0
    best_val_am_acc = 0.0
    best_epoch = 0
    best_val_loc = 0.0
    epoch_val_loss = np.array([0.0, 0.0, 0.0])

    epoch_val_dice = 0.0
    epoch_val_sen = 0.0
    epoch_val_pre = 0.0
    epoch_val_loc = 0.0
    epoch_val_mse = 1.0
    epoch_val_acc = 0.0


    best_model_path = r'\best_model-'
    model_iterative_path = os.path.join(model_save_path, 'iterative_input_model.pth')
    ema_decay = 0.99

    AM_Epoch = 0
    DM_Epoch = 0
    DM_VAL_Epoch = [999] #[65,162]
    AM_Memory_Bank = dict()
    # set TQDM iterator
    tqiter = trange(num_epochs, desc='GCN')
    scaler = GradScaler()
    use_amp = True
    iter_num = 0
    for epoch in tqiter:

        epoch_train_loss = np.array([0.0, 0.0, 0.0, 0.0])
        fig_loss = plt.figure(num='loss', figsize=[12, 3.8])

        # training_lr = learning_rate * (0.7 ** ((epoch - 0) // 30))  # 学习率衰减
        # for param_group in optimizer[0].param_groups:  # [{'amsgrad':False, 'betas':(0.9,0.99),'eps':1e-8,...}]
        #     param_group['lr'] = training_lr
        #
        # for param_group in optimizer[1].param_groups:  # [{'amsgrad':False, 'betas':(0.9,0.99),'eps':1e-8,...}]
        #     param_group['lr'] = training_lr

        training_lr_seg = optimizer[0].param_groups[0]['lr']
        training_lr_am = optimizer[1].param_groups[0]['lr']
        training_lr_dm = optimizer[2].param_groups[0]['lr']

        # go through all batches
        for i, sample1 in enumerate(modelDataLoader['trainLabeled']):
            if type(phase_now) == list:
                if i < (len(modelDataLoader['trainLabeled']) - 1) and epoch < AM_Epoch:
                    procedure = [phase_now[0]]
                elif i < (len(modelDataLoader['trainLabeled']) - 1) and epoch >= AM_Epoch:
                    procedure = [phase_now[1]]
                elif i >= (len(modelDataLoader['trainLabeled']) - 1) and epoch < AM_Epoch:
                    procedure = [phase_now[0],'val_labeled']
                else:
                    procedure = [phase_now[1], 'val_labeled']

            # if i < (len(modelDataLoader['trainLabeled']) - 1) and epoch < DM_Epoch:
            #     procedure = [phase_now[0]]
            # elif i < (len(modelDataLoader['trainLabeled']) - 1) and epoch >= DM_Epoch:
            #     procedure = [phase_now[1]]
            # elif i >= (len(modelDataLoader['trainLabeled']) - 1) and epoch < DM_Epoch:
            #     procedure = [phase_now[0],'val_labeled']
            # else:
            #     procedure = [phase_now[1], 'val_labeled']
            else:
                if i < (len(modelDataLoader['trainLabeled']) - 1):
                    procedure = [phase_now]
                else:
                    procedure = [phase_now, 'val_labeled']
            # run training and validation alternatively
            # run training and validation alternatively
            for phase in procedure:
                if phase == 'trainLabeled_seg':
                    if i == 0:
                        scheduler[0].step()
                    model.train()
                elif phase == 'trainLabeled_am':
                    if i == 0:
                        scheduler[1].step()
                    model.train()
                elif phase == 'trainLabeled_dm':
                    if i == 0:
                        scheduler[2].step()
                    model.train()
                elif phase == 'trainLabeled':
                    if i == 0:
                        scheduler[0].step()
                        scheduler[1].step()
                        scheduler[2].step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_loss_seg = 0.0
                running_loss_am = 0.0
                running_loss_dm = 0.0
                running_loss_con = 0.0
                running_loss_sup = 0.0
                loss_dm = 0.0
                # If 'labeled', then use segmentation mask; else use image for reconstruction
                case_name = sample1['case_name']
                inputs = sample1['image'].float().to(device)  # batch, FLAIR
                labels = sample1['mask'].float().to(device)
                labels_oc = sample1['mask_oc'].float().to(device)
                image = sample1['image'].float().to(device)
                mmip_mask = sample1['mmip_mask'].float().to(device)
                AM = sample1['am'].to(device)
                DM = sample1['distance_map'].float().to(device)
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                optimizer[2].zero_grad()

                # update model parameters and compute loss
                with torch.set_grad_enabled(
                        phase == 'trainLabeled' or phase == 'trainLabeled_seg' or phase == 'trainLabeled_am' or phase == 'trainLabeled_dm'):
                    if phase == 'trainLabeled_seg':
                        case_name = tuple(case_name)

                        w_sup = [1.0]
                        w_sdsc, w_sce, w_ssup = 1.0, 1.0, 1.0

                        with autocast(enabled=use_amp):
                            outputsL, _, fea_logit_set,_,_ = model(inputs, phase=phase, network_switch=network_switch,deep_supervision = True)
                            # outputsL ,_= model(inputs)

                            # up_fea_logit = [0,0,0]
                            # for fea_i in range(len(fea_logit_set)):
                            #     up_fea_logit[fea_i] = F.interpolate(fea_logit_set[fea_i], size=labels.size()[2:], mode='trilinear', align_corners=True)

                            # print(up_fea_logit.size(),fea_logit.size(),labels.size()[2:])
                            # print(labels_oc.shape)
                            loss_dsc = criterion[0](F.softmax(outputsL.float()), labels.float())
                            loss_ce = criterion[1](outputsL.float(), labels.float())
                            # loss_sup = w_sup[0] * criterion[1](up_fea_logit[0].float(), labels.float())
                                       # w_sup[1] * criterion[1](up_fea_logit[1].float(), labels_oc.long()) + \
                                       # w_sup[2] * criterion[1](up_fea_logit[2].float(), labels_oc.long())

                            # loss_seg = w_sdsc * loss_dsc + w_sce * loss_ce + w_ssup * loss_sup
                            loss_seg = w_sdsc * loss_dsc + w_sce * loss_ce
                            loss = loss_seg
                            # w1 * criterion[1](up_fea_logit.float(), labels.float())
                            # loss = w1 * criterion[0](outputsL.float(), labels.float())

                    if phase == 'trainLabeled_dm':
                        w1 = 1
                        outputsL, outputsL_am ,outputsL_dm= model(inputs, phase=phase, network_switch=network_switch)
                        # print(np.unique(outputsL_dm.cpu().detach().numpy()))
                        # print(outputsL_dm.squeeze().shape,DM.squeeze().shape)
                        loss = w1 * criterion[3](outputsL_dm.squeeze(), DM.squeeze())
                        # inputs_iterative = torch.concat((inputs[:,0,:].unsqueeze(1),outputsL_dm),dim=1).detach()
                        # print(inputs[:,0,:].unsqueeze(1).shape,outputsL_dm.shape,inputs_iterative.shape)
                        # dm_flag = True
                        torch.cuda.empty_cache()

                    elif phase == 'trainLabeled':
                        w_seg = 1.0
                        w_sup = [1.0]
                        w_sdsc,w_sce,w_ssup = 1.0,1.0,1.0

                        w_am = 1.0
                        w_adsc, w_ace = 1.0, 1.0

                        w_hcons = 0.5 #1.0
                        w_mse = 1.0


                        with autocast(enabled=use_amp):
                            outputsL, _, fea_logit_set,outputsL_am,_ = model(inputs, phase=phase, network_switch=network_switch,deep_supervision = True, mask = mmip_mask)
                            # outputsL ,_= model(inputs)
                            up_fea_logit = [0,0,0]
                            for fea_i in range(len(fea_logit_set)):
                                up_fea_logit[fea_i] = F.interpolate(fea_logit_set[fea_i], size=labels.size()[2:], mode='trilinear', align_corners=True)
                            # print(up_fea_logit.size(),fea_logit.size(),labels.size()[2:])
                            # print(labels_oc.shape)
                            outputsL_softmax = F.softmax(outputsL.float(),dim=1)
                            outputsL_am_softmax = F.softmax(outputsL_am.float(),dim=1)
                            # Seg
                            loss_dsc = criterion[0](outputsL_softmax, labels.float())
                            # loss_dsc = criterion[6](outputsL, labels.float())
                            # loss_ce = criterion[1](outputsL.float(), labels_oc.long())

                            loss_ce = criterion[1](outputsL.float(), labels.float())

                            loss_seg_sup = w_sup[0] * criterion[1](up_fea_logit[0].float(), labels_oc.long())
                                       # w_sup[1] * criterion[1](up_fea_logit[1].float(), labels_oc.long()) + \
                                       # w_sup[2] * criterion[1](up_fea_logit[2].float(), labels_oc.long())

                            loss_seg = w_sdsc * loss_dsc + w_sce * loss_ce + w_ssup * loss_seg_sup
                            # loss_seg = w_sdsc * loss_dsc + w_sce * loss_ce
                            # MMIP
                            loss_dsc_am = criterion[0](outputsL_am_softmax, mmip_mask.float())
                            loss_ce_am = criterion[1](outputsL_am.float(), mmip_mask.float())
                            loss_am =  w_ace * loss_ce_am +  w_adsc * loss_dsc_am

                            # Heterogeneous consistency
                            projectionsm_from_3d = ThreeD_Seg_to_TwoD_Proj(outputsL_softmax)

                            AM_threeDSeg = AM_Generate_Pred(projectionsm_from_3d)
                            AM_TwoDProj = AM_Generate_Pred_nograd(outputsL_am_softmax) #AM_Generate_Pred_nograd
                            # print(AM_threeDSeg,AM_TwoDProj)
                            # loss_GMSE = F.mse_loss(projectionsm_from_3d, outputsL_am_softmax)
                            loss_GMSE = criterion[4](AM_threeDSeg.unsqueeze(1),AM_TwoDProj.unsqueeze(1))

                            # loss_DscCon = criterion[0](projectionsm_from_3d, outputsL_am_softmax)
                            # loss_SSIM = criterion[5](AM_threeDSeg,AM_TwoDProj)
                            loss_hcons = w_mse * loss_GMSE
                            loss = w_seg * loss_seg  + w_am * loss_am + w_hcons * loss_hcons
                            # loss = w_seg * loss_seg + w_am * loss_am

                            # loss = w_seg * loss_seg  + w_am * loss_am
                        # loss = loss_am + loss_seg
                        # inputs_iterative = torch.concat((inputs[:,0,:].unsqueeze(1),outputsL_dm),dim=1).detach()

                    if phase == 'trainLabeled_am':

                        w_adsc, w_ace = 1.0, 1.0

                        with autocast(enabled=use_amp):
                            outputsL, _, fea_logit_set, outputsL_am, _ = model(inputs, phase=phase,
                                                                               network_switch=network_switch,
                                                                               deep_supervision=True)

                            # MMIP
                            loss_dsc_am = criterion[0](F.softmax(outputsL_am.float(), dim=1), mmip_mask.float())
                            loss_ce_am = criterion[1](outputsL_am.float(),mmip_mask.float())
                            loss_am = w_ace * loss_ce_am +  w_adsc * loss_dsc_am

                            loss = loss_am


                    if phase == 'trainLabeled_dm':
                        loss.backward() #retain_graph=True
                        optimizer[2].step()
                        running_loss += loss.item() * inputs.size(0)

                    if phase == 'trainLabeled':
                        if use_amp:
                            scaler.scale(loss).backward(retain_graph=True)
                            scaler.step(optimizer[0])
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer[0].step()

                        running_loss_seg += loss_seg.item() * inputs.size(0)
                        running_loss_am += loss_am.item() * inputs.size(0)
                        running_loss_con += loss_hcons.item() * inputs.size(0)
                        writer.add_scalar('seg loss', running_loss_seg, epoch)
                        writer.add_scalar('am loss', running_loss_am, epoch)
                        writer.add_scalar('con loss', running_loss_con, epoch)

                    if phase == 'trainLabeled_seg':
                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer[0])
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer[0].step()
                        running_loss += loss.item() * inputs.size(0)
                        # running_loss_seg += loss_dsc.item() * inputs.size(0)
                        # running_loss_am += loss_ce.item() * inputs.size(0)
                        # running_loss_sup += loss_sup.item() * inputs.size(0)

                    if phase == 'trainLabeled_am':
                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer[1])
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer[1].step()
                        running_loss += loss.item() * inputs.size(0)
                        # print(loss.item())

                # for ema_epoch_idx in range(len(DM_VAL_Epoch)):
                #     if epoch in range(DM_VAL_Epoch[ema_epoch_idx]-10,DM_VAL_Epoch[ema_epoch_idx]):
                #         update_ema_variables(model, model_copy, ema_decay, iter_num)
                #         # print('Updating EMA Variables')
                #         break

                iter_num += 1

                epoch_loss = running_loss
                epoch_loss_seg = running_loss_seg
                epoch_loss_am = running_loss_am
                epoch_loss_dm = running_loss_dm
                epoch_loss_con = running_loss_con
                # compute loss
                if phase == 'trainLabeled_seg':
                    # epoch_train_loss[0] += epoch_loss
                    epoch_train_loss[0] += epoch_loss

                elif phase == 'trainLabeled_am':
                    epoch_train_loss[1] += epoch_loss
                elif phase == 'trainLabeled_dm':
                    epoch_train_loss[2] += epoch_loss
                elif phase == 'trainLabeled':
                    epoch_train_loss[0] += epoch_loss_seg
                    epoch_train_loss[1] += epoch_loss_am
                    epoch_train_loss[2] += epoch_loss_con


                # compute validation accuracy, update training and validation loss, and calculate DICE and MSE
                if (epoch + 1) in [1,50,60,150,160,280,290,300]: #epoch % 10 == 9:
                    if phase == 'val_labeled':
                        if epoch < AM_Epoch:
                            running_val_acc,epoch_val_loss[1], Test_Memory_Bank = eval_net_AM(model, model_copy,criterion, phase, network_switch, modelDataLoader['val_labeled'],
                                                            preview=PREVIEW, gpu=True, visualize_batch=0, epoch=epoch, slice=18, root_path=root_path,AM_decoder = AM_decoder,DM_VAL_Epoch = DM_VAL_Epoch,model_iterative_path = model_iterative_path)
                            epoch_val_acc = running_val_acc
                        # elif epoch < DM_Epoch:
                        #     running_val_loc, epoch_val_loss[2] = eval_net_hm(model, model_copy, criterion, phase, network_switch, modelDataLoader['val_labeled'],
                        #                                 preview=PREVIEW, gpu=True,visualize_batch=0, epoch=epoch, slice=18,root_path=root_path,DM_VAL_Epoch = DM_VAL_Epoch,model_iterative_path = model_iterative_path)
                        #     epoch_val_loc = running_val_loc
                        else:
                            # running_val_loc, epoch_val_loss[2] = eval_net_hm(model, model_copy, criterion, phase, network_switch, modelDataLoader['val_labeled'],
                            #                             preview=PREVIEW, gpu=True,visualize_batch=0, epoch=epoch, slice=18,root_path=root_path,DM_VAL_Epoch = DM_VAL_Epoch,model_iterative_path = model_iterative_path)
                            # epoch_val_loc = running_val_loc
                            # running_val_acc, epoch_val_loss[1], Test_Memory_Bank = eval_net_AM(model, model_copy, criterion, phase, network_switch,modelDataLoader['val_labeled'],
                            #                                 preview=PREVIEW, gpu=True,visualize_batch=0, epoch=epoch, slice=18,root_path=root_path,AM_decoder = AM_decoder,DM_VAL_Epoch = DM_VAL_Epoch,model_iterative_path = model_iterative_path)
                            # epoch_val_acc = running_val_acc

                            # running_val_dice,running_val_sen,running_val_pre,epoch_val_loss[0] = eval_net_dice(model, criterion, phase,network_switch, modelDataLoader['val_labeled'],
                            #                                 preview=PREVIEW, gpu=True, visualize_batch=0, epoch=epoch, slice=18, root_path=root_path)
                            # epoch_val_dice,epoch_val_sen,epoch_val_pre = running_val_dice,running_val_sen,running_val_pre

                            running_val_dice,running_val_sen,running_val_pre,running_val_acc,epoch_val_loss[0] = eval_net_dice_am(model, model_copy, criterion, phase,network_switch, modelDataLoader['val_labeled'],
                                                            preview=PREVIEW, gpu=True, visualize_batch=0, epoch=epoch, slice=18, root_path=root_path,DM_VAL_Epoch = DM_VAL_Epoch,model_iterative_path = model_iterative_path)
                            epoch_val_dice,epoch_val_sen,epoch_val_pre,epoch_val_acc = running_val_dice,running_val_sen,running_val_pre,running_val_acc

                            writer.add_scalar('val dice', epoch_val_dice, epoch)
                            writer.add_scalar('val sen', epoch_val_sen, epoch)
                            writer.add_scalar('val pre', epoch_val_pre, epoch)
                            writer.add_scalar('val acc', epoch_val_acc, epoch)
                            # running_val_dice,running_val_sen,running_val_pre,epoch_val_loss[0] = eval_net_dice(model, criterion, phase, network_switch, modelDataLoader['val_labeled'],
                            #                             preview=PREVIEW, gpu=True, visualize_batch=0, epoch=epoch, slice=18, root_path=root_path)
                            # epoch_val_dice,epoch_val_sen,epoch_val_pre = running_val_dice,running_val_sen,running_val_pre
                        # running_val_huber, epoch_val_loss[1] = eval_net_hm(model, criterion, phase, network_switch, modelDataLoader['val_labeled'],
                        #                             preview=PREVIEW, gpu=True,visualize_batch=0, epoch=epoch, slice=18,root_path=root_path)
                        # epoch_val_huber = running_val_huber

                # # display TQDM information
                tqiter.set_description('GCN (SEG=%.4f, AM=%.4f, HCON=%.4f, vam_acc=%.4f, vdice=%.4f, vsen=%.4f, vpre=%.4f, vloc=%.4f, seg lr=%f, am lr=%f, dm lr=%f)'
                                       % (epoch_train_loss[0]/(2*(i+1)), epoch_train_loss[1]/(2*(i+1)), epoch_train_loss[2]/(2*(i+1)),
                                          epoch_val_acc,epoch_val_dice, epoch_val_sen,epoch_val_pre,epoch_val_loc,training_lr_seg,training_lr_am,training_lr_dm))

                # save best validation model, figure preview and dice
                if phase == 'val_labeled':
                    if epoch < AM_Epoch and ((epoch_val_acc > best_val_am_acc)):
                        if os.path.exists(best_model_path) and epoch_val_acc < 0.85:
                            os.remove(best_model_path)
                        best_epoch = epoch
                        best_val_am_acc = epoch_val_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                        best_model_path = os.path.join(model_save_path,
                                                       f'best_model-{best_epoch + 1}-' + '{0:.4f}.pth'.format(best_val_am_acc))
                        torch.save(model.state_dict(), best_model_path)

                    if epoch < DM_Epoch and ((epoch_val_loc > best_val_loc)):
                        if os.path.exists(best_model_path) and epoch_val_loc < 0.85:
                            os.remove(best_model_path)
                        best_epoch = epoch
                        best_val_loc = epoch_val_loc
                        best_model_wts = copy.deepcopy(model.state_dict())

                        best_model_path = os.path.join(model_save_path,
                                                       f'best_model-{best_epoch + 1}-' + '{0:.4f}.pth'.format(best_val_loc))
                        torch.save(model.state_dict(), best_model_path)

                    elif epoch >= AM_Epoch and ((epoch_val_dice > best_val_dice) or (epoch_val_sen > best_val_sensitivity) or (epoch_val_pre > best_val_precision)):
                        if os.path.exists(best_model_path) and epoch_val_sen < 0.7 and epoch_val_pre < 0.7 :
                            os.remove(best_model_path)
                        best_epoch = epoch
                        best_val_dice = epoch_val_dice
                        best_val_sensitivity = epoch_val_sen
                        best_val_precision = epoch_val_pre
                        best_val_am_acc = epoch_val_acc
                        best_val_loc = epoch_val_loc
                        best_model_wts = copy.deepcopy(model.state_dict())

                        best_model_path = os.path.join(model_save_path,
                                                       f'best_model-{best_epoch + 1}-' + '{0:.4f}-{1:.4f}-{2:.4f}-{3:.4f}-{4:.4f}.pth'.format(
                                                           best_val_dice, best_val_sensitivity, best_val_precision,
                                                           best_val_am_acc,best_val_loc))
                        torch.save(model.state_dict(), best_model_path)

        if (epoch+1) in DM_VAL_Epoch:
                model_path = os.path.join(model_save_path,f'iterative_input_model.pth')
                torch.save(model_copy.state_dict(), model_path)
                print('Ready to iterative...')
        # if epoch > 100 and (epoch + 1) % 20 == 0:  # 10次保存一次
        #     torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch-{epoch + 1}' + '.pth'))
        #     print('Model {} saved.'.format(epoch + 1))

    # compute run time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Dice: {:4f}'.format(best_val_dice))
    print('Best val Sensitivity: {:4f}'.format(best_val_sensitivity))
    print('Best val Precision: {:4f}'.format(best_val_precision))
    print('Best val AM acc: {:4f}'.format(best_val_am_acc))
    model.load_state_dict(best_model_wts)
    return model, best_val_dice


# Set up training
def network_training_epoch(Test_only, job, data_seed, data_split, device, data_sizes, modelDataLoader, num_class, num_epoch,shape, folder_name, TSNE, GNN_model,SEG_AM_DM,checkpoint_dir, model_load_path,model_load):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    val_dice = 0
    test_results = 0

    device = device
    dataset_sizes = data_sizes

    print('-' * 64)
    print('Training start')

    basic_path = folder_name +'/' + str(data_split)[:]

    #################################################
    if job == 'GCN':

        switch = {'trainL_encoder': True,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_am': True,
                  'trainL_decoder_dm': True}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        if not os.path.exists(os.path.join(basic_path,'result')):
            os.mkdir(os.path.join(basic_path,'result'))

        base_features = 32
        AM_edge_num = int((num_class + 1) * num_class / 2)

        if SEG_AM_DM[2] == '0':
            in_channels = 1
        else:
            in_channels = 2

        model = TSG_GCN.GCN_v1(in_channels, num_class, AM_edge_num,base_features,Seg = SEG_AM_DM[0],AM = SEG_AM_DM[1], DM = SEG_AM_DM[2],gnn_model = GNN_model).to(device)

        print(model)
        print_model_parm_nums(model)

        if SEG_AM_DM == ['0','0','1']:
            phase_now = 'trainLabeled_dm'
        elif SEG_AM_DM == ['0','1','0']:
            phase_now = 'trainLabeled_am'
        elif SEG_AM_DM == ['1','0','0']:
            phase_now = 'trainLabeled_seg'
        elif SEG_AM_DM == ['1','1','0']:
            phase_now = ['trainLabeled_am','trainLabeled']
        elif SEG_AM_DM == ['1','1','1']:
            phase_now = 'trainLabeled'
        else:
            phase_now = 'trainLabeled'
        print(phase_now)

        # if not os.path.exists(os.path.join(root_path,'model/Comparsion')):
        #     os.mkdir(os.path.join(root_path,'model/Comparsion'))

        checkpoint_dir = os.path.join(root_path, checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model_save_path = checkpoint_dir + '/' + 'save/'
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        model_load_path = os.path.join(root_path,model_load_path)
        save_path = 'result_' + model_load_path.split('/')[-1]


        if not os.path.exists(os.path.join(basic_path, 'result', save_path)):
            # print(os.path.join(basic_path, 'result', save_path),basic_path,save_path)
            os.mkdir(os.path.join(basic_path, 'result', save_path))


        writer = SummaryWriter(fr'{checkpoint_dir}/logs')

        if os.path.exists(model_load_path) and model_load:
            model.load_state_dict(torch.load(model_load_path))
            print('Model loaded from {}.'.format(model_load_path))
        else:
            model.apply(TSG_GCN.weights_init)
            # model.apply(ssl_3d_GCN_DM_related_jointly_deep_supervision.InitWeights_He(1e-2))
            print('Building a new model...')

            # torch.save(model.state_dict(), '{0}/initial.pth'.format(model_save_path))

        # model_copy = deepcopy(model)
        # for param in model_copy.parameters():
        #     param.detach_()
        # model_copy.eval()

        weight = torch.ones(num_class).cuda()
        if num_class == 16:
            weight[9], weight[10], weight[14] = 10, 10, 10
        # print('Weight CE',weight)
        if not Test_only:
            criterionDICE = DiceLoss() #DiceCoefficientLF(device)
            criterionCE = nn.CrossEntropyLoss(weight=weight) #nn.CrossEntropyLoss() #DC_and_CE_loss({'batch_dice': 2, 'smooth': 1e-5, 'do_bg': False}, {}) #crossentropy()
            criterionBCE = torch.nn.BCEWithLogitsLoss()
            criterionHuber = Huber_loss()
            criterionMSE = Guassian_MSE()
            criterionSSIM = cosine_similarity() #SSIM() #Guassian_MSE()# cosine_similarity() #nn.MSELoss(reduction='mean')
            criterionWCE = WeightedCE()
            criterion = (criterionDICE, criterionCE, criterionBCE,criterionHuber,criterionMSE,criterionSSIM,criterionWCE)

            learning_rate = 1e-3

            optimizer_ft = (optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005),
                            optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005),
                            optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005))

            # exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=256, gamma=0.7),
            #                     lr_scheduler.StepLR(optimizer_ft[1], step_size=256, gamma=0.7))

            exp_lr_scheduler = (CosineAnnealingLRWithRestarts(optimizer_ft[0], T_0=65, T_mult=1.5, eta_min=0.00001, k=0.75),
                                CosineAnnealingLRWithRestarts(optimizer_ft[1], T_0=65, T_mult=1.5, eta_min=0.00001, k=0.75),
                                CosineAnnealingLRWithRestarts(optimizer_ft[2], T_0=65, T_mult=1.5, eta_min=0.00001, k=0.75))
                                # lr_scheduler.StepLR(optimizer_ft[2], step_size=500, gamma=0.5))
            # save training information
            train_info = (
                'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\n'
                'base features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
                'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                    job,
                    data_seed,
                    data_split,
                    dataset_sizes,
                    type(model),
                    base_features,
                    switch,
                    criterion,
                    optimizer_ft,
                    exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model,0, modelDataLoader, model_save_path,writer,device, root_path, switch, criterion, optimizer_ft,
                                          exp_lr_scheduler,learning_rate,phase_now,num_epochs=num_epoch, loss_weighted=True)

            # Testing model
            test_results = test_net_dice(model_load_path, save_path, basic_path, model, switch, modelDataLoader['test'],
                                         num_class, shape, TSNE, gpu=True)
            print('GCN finished')

        else:
            test_results = test_net_dice_w_2dproj(model_load_path, save_path, basic_path, model, switch, modelDataLoader['test'],
                                         num_class, shape, TSNE, gpu=True)
            print('GCN finished')

    return val_dice, test_results