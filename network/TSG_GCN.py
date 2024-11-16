import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from network.Graph_Extractor_Module import GBA
import scipy.sparse as sp
import numpy as np
from dataloader.utils import check_symmetric,normalize_adj

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform(m.weight.data)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class GCN_v1(nn.Module):
    def __init__(self, in_channels, n_classes, AM_out_channels = 55,base_n_filter=32,Seg = 1,AM = 1, DM = 1,gnn_model = 'GCN'):
        super(GCN_v1, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.Seg_only = Seg
        self.AM_only = AM
        self.DM_only = DM
        self.gnn_model = gnn_model
        self.encoder1 = GCN_v1._block(in_channels, base_n_filter, name="enc1")
        self.e_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = GCN_v1._block(base_n_filter, base_n_filter * 2, name="enc2")
        self.e_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = GCN_v1._block(base_n_filter * 2, base_n_filter * 4, name="enc3")
        self.e_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = GCN_v1._block(base_n_filter * 4, base_n_filter * 8, name="enc4")
        self.e_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck_seg_2 = GCN_v1._block(base_n_filter * 8, (base_n_filter * 8) * 2, name="bottleneck")

        if int(self.AM_only) == 1:
            self.am_pool1 = nn.AdaptiveMaxPool3d((1, 160, 160))
            self.am_pool2 = nn.AdaptiveMaxPool3d((1, 80, 80))
            self.am_pool3 = nn.AdaptiveMaxPool3d((1, 40, 40))
            self.am_pool4 = nn.AdaptiveMaxPool3d((1, 20, 20))
            self.am_pool_bn = nn.AdaptiveMaxPool3d((1, 10, 10))

            self.upconv4_am = nn.ConvTranspose2d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4_am = GCN_v1._2Dblock((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4_am")
            self.upconv3_am = nn.ConvTranspose2d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3_am = GCN_v1._2Dblock((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3_am")
            self.upconv2_am = nn.ConvTranspose2d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2_am = GCN_v1._2Dblock((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2_am")
            self.upconv1_am = nn.ConvTranspose2d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1_am = GCN_v1._2Dblock(base_n_filter * 2, base_n_filter, name="dec1_am")

            self.conv_am = nn.Conv2d(base_n_filter, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        if int(self.Seg_only) == 1:
            # self.bottleneck_seg_2 = GBA(base_n_filter * 8, (base_n_filter * 8) * 2,class_num=n_classes)
            # self.bottleneck_seg_2 = GBA(base_n_filter * 8, class_num=n_classes,gnn_model = self.gnn_model)

            self.upconv4 = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4 = GCN_v1._block((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4")
            self.conv1_4 = GCN_v1._singleblock(base_n_filter * 8,base_n_filter, name="conv1_4")
            # self.decoder4 = GBA((base_n_filter * 8) * 2, base_n_filter * 8, class_num=n_classes)
            self.upconv3 = nn.ConvTranspose3d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3 = GCN_v1._block((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3")
            self.conv1_3 = GCN_v1._singleblock(base_n_filter * 4, base_n_filter * 2, name="conv1_3")
            # self.decoder3 = GBA((base_n_filter * 4) * 2, base_n_filter * 4, class_num=n_classes)
            self.upconv2 = nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2 = GCN_v1._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2")
            self.conv1_2 = GCN_v1._singleblock(base_n_filter * 2, base_n_filter , name="conv1_2")

            self.decoder3_high_level = GCN_v1._singleblock(base_n_filter * 4,(base_n_filter * 4), name="dec3_high_level")
            self.decoder3_GCN = GBA(input_num=base_n_filter * 4, output_channels=base_n_filter * 4,hidden_layer=base_n_filter * 4, class_num=n_classes)
            # self.decoder3_GCN = GCN_v1._singleblock((base_n_filter * 2) * 2, base_n_filter * 4, name="dec3_GCN")

            self.upconv2_GCN = nn.ConvTranspose3d(base_n_filter * 4 * 2, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder2_GCN = GCN_v1._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2_GCN")

            self.upconv1 = nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1 = GCN_v1._block(base_n_filter * 2, base_n_filter,  name="dec1")

            self.conv = nn.Conv3d(in_channels=base_n_filter, out_channels=n_classes, kernel_size=1)

        if int(self.DM_only) == 1:
            self.upconv4_loc = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4_loc = GCN_v1._block((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4")
            self.upconv3_loc = nn.ConvTranspose3d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3_loc = GCN_v1._block((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3")
            self.upconv2_loc = nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2_loc = GCN_v1._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2")
            self.upconv1_loc = nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1_loc = GCN_v1._block(base_n_filter * 2, base_n_filter, name="dec1")

            self.conv_loc = nn.Conv3d(in_channels=base_n_filter, out_channels=1, kernel_size=1)


    def forward(self, x, phase, network_switch,deep_supervision = False,mask = None):
        if phase != 'val_labeled':
            if phase == 'trainLabeled':
                encoder = True
                decoder_seg = True
                decoder_am = True
            if phase == 'trainLabeled_seg':
                encoder = True
                decoder_seg = True
                decoder_am = False
            if phase == 'trainLabeled_am':
                encoder = True
                decoder_seg = False
                decoder_am = True
            if phase == 'trainLabeled_am':
                encoder = True
                decoder_seg = False
                decoder_am = True

            # Set trainable parameters for Shared Encoder
            for param in self.encoder1.parameters():
                param.requires_grad = encoder
            for param in self.encoder2.parameters():
                param.requires_grad = encoder
            for param in self.encoder3.parameters():
                param.requires_grad = encoder
            for param in self.encoder4.parameters():
                param.requires_grad = encoder

            for param in self.e_pool1.parameters():
                param.requires_grad = encoder
            for param in self.e_pool2.parameters():
                param.requires_grad = encoder
            for param in self.e_pool3.parameters():
                param.requires_grad = encoder
            for param in self.e_pool4.parameters():
                param.requires_grad = encoder
            for param in self.bottleneck_seg_2.parameters():
                param.requires_grad = decoder_seg

            # MLP AM
            if int(self.AM_only) == 1:
                for param in self.am_pool1.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool2.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool3.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool4.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool_bn.parameters():
                    param.requires_grad = decoder_am

                for param in self.upconv4_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.upconv3_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.upconv2_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.upconv1_am.parameters():
                    param.requires_grad = decoder_am

                for param in self.decoder4_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.decoder3_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.decoder2_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.decoder1_am.parameters():
                    param.requires_grad = decoder_am

                for param in self.conv_am.parameters():
                    param.requires_grad = decoder_am

            if int(self.Seg_only) == 1:
                for param in self.upconv4.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv3.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv2.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv2_GCN.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv1.parameters():
                    param.requires_grad = decoder_seg

                for param in self.decoder1.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder2.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder2_GCN.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder3.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder3_GCN.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder3_high_level.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder4.parameters():
                    param.requires_grad = decoder_seg
                for param in self.conv.parameters():
                    param.requires_grad = decoder_seg

                for param in self.conv1_2.parameters():
                    param.requires_grad = decoder_seg
                for param in self.conv1_3.parameters():
                    param.requires_grad = decoder_seg
                for param in self.conv1_4.parameters():
                    param.requires_grad = decoder_seg

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.e_pool1(enc1))
        enc3 = self.encoder3(self.e_pool2(enc2))
        enc4 = self.encoder4(self.e_pool3(enc3))
        bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4))

        if int(self.DM_only) == 1:
            bottleneck_loc = self.bottleneck_loc(self.e_pool4(enc4))
            # bottleneck_seg = torch.cat((bottleneck_seg, bottleneck_AM), dim=1)
            # bottleneck_seg = self.decoder_joint(bottleneck_seg)
            # for param in self.bottleneck_loc.parameters():
            #     print('bottleneck_loc', param.requires_grad)
            dec4_loc = self.upconv4_loc(bottleneck_loc)
            # for param in self.upconv4.parameters():
            #     print('dec4', param.requires_grad)
            dec4_loc = torch.cat((dec4_loc, enc4), dim=1)
            dec4_loc = self.decoder4_loc(dec4_loc)
            dec3_loc = self.upconv3_loc(dec4_loc)
            dec3_loc = torch.cat((dec3_loc, enc3), dim=1)
            dec3_loc = self.decoder3_loc(dec3_loc)
            dec2_loc = self.upconv2_loc(dec3_loc)
            dec2_loc = torch.cat((dec2_loc, enc2), dim=1)
            dec2_loc = self.decoder2_loc(dec2_loc)
            dec1_loc = self.upconv1_loc(dec2_loc)
            dec1_loc = torch.cat((dec1_loc, enc1), dim=1)
            dec1_loc = self.decoder1_loc(dec1_loc)
            out_loc = self.conv_loc(dec1_loc)
        else:
            out_loc = torch.zeros_like(x)

        if int(self.AM_only) == 1:
            am_out1 = self.am_pool_bn(bottleneck_seg)
            b, c, _, w, d = am_out1.shape
            am_out1 = am_out1.reshape(b, c, w, d)
            dec4_am = self.upconv4_am(am_out1)
            dec4_am = torch.cat((dec4_am, self.am_pool4(enc4).reshape(dec4_am.shape)), dim=1)
            dec4_am = self.decoder4_am(dec4_am)
            dec3_am = self.upconv3_am(dec4_am)
            dec3_am = torch.cat((dec3_am, self.am_pool3(enc3).reshape(dec3_am.shape)), dim=1)
            dec3_am = self.decoder3_am(dec3_am)
            dec2_am = self.upconv2_am(dec3_am)
            dec2_am = torch.cat((dec2_am, self.am_pool2(enc2).reshape(dec2_am.shape)), dim=1)
            dec2_am = self.decoder2_am(dec2_am)
            dec1_am = self.upconv1_am(dec2_am)
            dec1_am = torch.cat((dec1_am, self.am_pool1(enc1).reshape(dec1_am.shape)), dim=1)
            dec1_am = self.decoder1_am(dec1_am)
            out_AM = self.conv_am(dec1_am)
        else:
            out_AM = torch.zeros_like(x)

        if int(self.Seg_only) == 1:
            # if deep_supervision:
            #     bottleneck_seg,out_cls,mask_cls = self.bottleneck_seg_2(self.e_pool4(enc4),edge_indexs,deep_supervision,mask)
            #     # for param in self.bottleneck_seg_2.parameters():
            #     #     print('bottleneck_seg_2', param.requires_grad)
            # else:
            #     # bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4), edge_indexs) # GCN
            #     bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4))  # GCN
            #     # bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4)) # CNN
            # # bottleneck_seg = torch.cat((bottleneck_seg, bottleneck_AM), dim=1)
            # # bottleneck_seg = self.decoder_joint(bottleneck_seg)


            dec4 = self.upconv4(bottleneck_seg)
            # for param in self.upconv4.parameters():
            #     print('dec4', param.requires_grad)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)

            dec4_fuse = self.conv1_4(dec4)
            dec3_fuse = self.conv1_3(dec3)
            dec2_fuse = self.conv1_2(dec2)
            dec4_fuse = F.interpolate(dec4_fuse, size=dec3.shape[2:], mode="trilinear", align_corners=True)
            dec2_fuse = F.interpolate(dec2_fuse, size=dec3.shape[2:], mode="trilinear", align_corners=True)
            gcn_dec = torch.cat((dec4_fuse, dec2_fuse, dec3_fuse), dim=1)
            # print(gcn_dec.shape)
            gcn_high_level = self.decoder3_high_level(gcn_dec)

            if int(self.AM_only) == 1:
                gcn_dec, fea_logit = self.decoder3_GCN(gcn_dec,F.softmax(out_AM,1))
            else:
                gcn_dec,fea_logit = self.decoder3_GCN(gcn_dec)
                # gcn_dec = self.decoder3_GCN(gcn_dec)
                # fea_logit = torch.zeros_like(gcn_dec)
            # gcn_combine = gcn_high_level + gcn_dec
            gcn_combine = torch.cat((gcn_high_level,gcn_dec), dim=1)

            # print(gcn_dec.shape)
            dec2_GCN = self.upconv2_GCN(gcn_combine)
            # print(dec2_GCN.shape)
            dec2_GCN = self.decoder2_GCN(dec2_GCN)
            # dec2_Combine = torch.cat((dec2_GCN, dec2), dim=1)


            dec1 = self.upconv1(dec2_GCN)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            out_seg = self.conv(dec1)
            fea_logit_display = F.interpolate(fea_logit, size=out_seg.shape[2:], mode="trilinear", align_corners=True)
        else:
            out_seg = torch.zeros_like(x)
            fea_logit_display = torch.zeros_like(x)
            fea_logit = torch.zeros_like(x)


        if deep_supervision:
            return out_seg,fea_logit_display,[fea_logit],out_AM,out_loc
        else:
            if self.n_classes == 1:
                return F.sigmoid(out_seg),out_AM,out_loc
            else:
                return out_seg,fea_logit,out_AM,out_loc

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ]
            )
        )
    @staticmethod
    def _singleblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                ]
            )
        )
    @staticmethod
    def _conv1block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                ]
            )
        )



    @staticmethod
    def _2Dblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ]
            )
        )

class GCN_v0(nn.Module):
    def __init__(self, in_channels, n_classes, AM_out_channels = 55,base_n_filter=32,Seg = 1,AM = 1, DM = 1,gnn_model = 'GCN'):
        super(GCN_v0, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.Seg_only = Seg
        self.AM_only = AM
        self.DM_only = DM
        self.gnn_model = gnn_model
        self.encoder1 = GCN_v0._block(in_channels, base_n_filter, name="enc1")
        self.e_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = GCN_v0._block(base_n_filter, base_n_filter * 2, name="enc2")
        self.e_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = GCN_v0._block(base_n_filter * 2, base_n_filter * 4, name="enc3")
        self.e_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = GCN_v0._block(base_n_filter * 4, base_n_filter * 8, name="enc4")
        self.e_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck_seg_2 = GCN_v0._block(base_n_filter * 8, (base_n_filter * 8) * 2, name="bottleneck")

        if int(self.AM_only) == 1:
            self.am_pool1 = nn.AdaptiveMaxPool3d((1, 160, 160))
            self.am_pool2 = nn.AdaptiveMaxPool3d((1, 80, 80))
            self.am_pool3 = nn.AdaptiveMaxPool3d((1, 40, 40))
            self.am_pool4 = nn.AdaptiveMaxPool3d((1, 20, 20))
            self.am_pool_bn = nn.AdaptiveMaxPool3d((1, 10, 10))

            self.upconv4_am = nn.ConvTranspose2d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4_am = GCN_v0._2Dblock((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4_am")
            self.upconv3_am = nn.ConvTranspose2d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3_am = GCN_v0._2Dblock((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3_am")
            self.upconv2_am = nn.ConvTranspose2d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2_am = GCN_v0._2Dblock((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2_am")
            self.upconv1_am = nn.ConvTranspose2d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1_am = GCN_v0._2Dblock(base_n_filter * 2, base_n_filter, name="dec1_am")

            self.conv_am = nn.Conv2d(base_n_filter, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        if int(self.Seg_only) == 1:
            # self.bottleneck_seg_2 = GBA(base_n_filter * 8, (base_n_filter * 8) * 2,class_num=n_classes)
            # self.bottleneck_seg_2 = GBA(base_n_filter * 8, class_num=n_classes,gnn_model = self.gnn_model)

            self.upconv4 = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4 = GCN_v0._block((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4")

            self.decoder4_high_level = GCN_v0._singleblock(base_n_filter * 8,(base_n_filter * 8), name="bottleneck_high_level")
            self.decoder4_GCN = GBA(input_num=base_n_filter * 8, output_channels=base_n_filter * 8,hidden_layer=base_n_filter * 8, class_num=n_classes)

            # self.decoder4 = GBA((base_n_filter * 8) * 2, base_n_filter * 8, class_num=n_classes)
            self.upconv3 = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3 = GCN_v0._block((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3")
            # self.decoder3 = GBA((base_n_filter * 4) * 2, base_n_filter * 4, class_num=n_classes)
            self.upconv2 = nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2 = GCN_v0._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2")

            self.upconv1 = nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1 = GCN_v0._block(base_n_filter * 2, base_n_filter,  name="dec1")

            self.conv = nn.Conv3d(in_channels=base_n_filter, out_channels=n_classes, kernel_size=1)

        if int(self.DM_only) == 1:
            self.upconv4_loc = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4_loc = GCN_v0._block((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4")
            self.upconv3_loc = nn.ConvTranspose3d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3_loc = GCN_v0._block((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3")
            self.upconv2_loc = nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2_loc = GCN_v0._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2")
            self.upconv1_loc = nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1_loc = GCN_v0._block(base_n_filter * 2, base_n_filter, name="dec1")

            self.conv_loc = nn.Conv3d(in_channels=base_n_filter, out_channels=1, kernel_size=1)


    def forward(self, x, phase, network_switch,deep_supervision = False,mask = None):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.e_pool1(enc1))
        enc3 = self.encoder3(self.e_pool2(enc2))
        enc4 = self.encoder4(self.e_pool3(enc3))
        bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4))


        out_loc = torch.zeros_like(x)

        if int(self.AM_only) == 1:
            am_out1 = self.am_pool_bn(bottleneck_seg)
            b, c, _, w, d = am_out1.shape
            am_out1 = am_out1.reshape(b, c, w, d)
            dec4_am = self.upconv4_am(am_out1)
            dec4_am = torch.cat((dec4_am, self.am_pool4(enc4).reshape(dec4_am.shape)), dim=1)
            dec4_am = self.decoder4_am(dec4_am)
            dec3_am = self.upconv3_am(dec4_am)
            dec3_am = torch.cat((dec3_am, self.am_pool3(enc3).reshape(dec3_am.shape)), dim=1)
            dec3_am = self.decoder3_am(dec3_am)
            dec2_am = self.upconv2_am(dec3_am)
            dec2_am = torch.cat((dec2_am, self.am_pool2(enc2).reshape(dec2_am.shape)), dim=1)
            dec2_am = self.decoder2_am(dec2_am)
            dec1_am = self.upconv1_am(dec2_am)
            dec1_am = torch.cat((dec1_am, self.am_pool1(enc1).reshape(dec1_am.shape)), dim=1)
            dec1_am = self.decoder1_am(dec1_am)
            out_AM = self.conv_am(dec1_am)
        else:
            out_AM = torch.zeros_like(x)

        if int(self.Seg_only) == 1:
            dec4 = self.upconv4(bottleneck_seg)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)

            gcn_high_level = self.decoder4_high_level(dec4)
            if int(self.AM_only) == 1:
                gcn_dec, fea_logit = self.decoder4_GCN(dec4, F.softmax(out_AM, 1))
            else:
                gcn_dec, fea_logit = self.decoder4_GCN(dec4)
            gcn_combine = torch.cat((gcn_high_level, gcn_dec), dim=1)

            dec3 = self.upconv3(gcn_combine)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            out_seg = self.conv(dec1)
            fea_logit_display = F.interpolate(fea_logit, size=out_seg.shape[2:], mode="trilinear", align_corners=True)
        else:
            out_seg = torch.zeros_like(x)
            fea_logit_display = torch.zeros_like(x)
            fea_logit = torch.zeros_like(x)


        if deep_supervision:
            return out_seg,fea_logit_display,[fea_logit],out_AM,out_loc
        else:
            if self.n_classes == 1:
                return F.sigmoid(out_seg),out_AM,out_loc
            else:
                return out_seg,fea_logit,out_AM,out_loc

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ]
            )
        )
    @staticmethod
    def _singleblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                ]
            )
        )
    @staticmethod
    def _conv1block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                ]
            )
        )



    @staticmethod
    def _2Dblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ]
            )
        )


class GCN_v2(nn.Module):
    def __init__(self, in_channels, n_classes, AM_out_channels=55, base_n_filter=32, Seg=1, AM=1, DM=1,
                 gnn_model='GCN'):
        super(GCN_v2, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.Seg_only = Seg
        self.AM_only = AM
        self.DM_only = DM
        self.gnn_model = gnn_model
        self.encoder1 = GCN_v2._block(in_channels, base_n_filter, name="enc1")
        self.e_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = GCN_v2._block(base_n_filter, base_n_filter * 2, name="enc2")
        self.e_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = GCN_v2._block(base_n_filter * 2, base_n_filter * 4, name="enc3")
        self.e_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = GCN_v2._block(base_n_filter * 4, base_n_filter * 8, name="enc4")
        self.e_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck_seg_2 = GCN_v2._block(base_n_filter * 8, (base_n_filter * 8) * 2, name="bottleneck")

        if int(self.AM_only) == 1:
            self.am_pool1 = nn.AdaptiveMaxPool3d((1, 160, 160))
            self.am_pool2 = nn.AdaptiveMaxPool3d((1, 80, 80))
            self.am_pool3 = nn.AdaptiveMaxPool3d((1, 40, 40))
            self.am_pool4 = nn.AdaptiveMaxPool3d((1, 20, 20))
            self.am_pool_bn = nn.AdaptiveMaxPool3d((1, 10, 10))

            self.upconv4_am = nn.ConvTranspose2d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4_am = GCN_v2._2Dblock((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4_am")
            self.upconv3_am = nn.ConvTranspose2d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3_am = GCN_v2._2Dblock((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3_am")
            self.upconv2_am = nn.ConvTranspose2d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2_am = GCN_v2._2Dblock((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2_am")
            self.upconv1_am = nn.ConvTranspose2d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1_am = GCN_v2._2Dblock(base_n_filter * 2, base_n_filter, name="dec1_am")

            self.conv_am = nn.Conv2d(base_n_filter, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        if int(self.Seg_only) == 1:
            # self.bottleneck_seg_2 = GBA(base_n_filter * 8, (base_n_filter * 8) * 2,class_num=n_classes)
            # self.bottleneck_seg_2 = GBA(base_n_filter * 8, class_num=n_classes,gnn_model = self.gnn_model)
            self.conv1_b = GCN_v2._singleblock(base_n_filter * 16, base_n_filter, name="conv1_4")

            self.upconv4 = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4 = GCN_v2._block((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4")
            self.conv1_4 = GCN_v2._singleblock(base_n_filter * 8, base_n_filter, name="conv1_4")
            # self.decoder4 = GBA((base_n_filter * 8) * 2, base_n_filter * 8, class_num=n_classes)
            self.upconv3 = nn.ConvTranspose3d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3 = GCN_v2._block((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3")
            self.conv1_3 = GCN_v2._singleblock(base_n_filter * 4, base_n_filter , name="conv1_3")
            # self.decoder3 = GBA((base_n_filter * 4) * 2, base_n_filter * 4, class_num=n_classes)
            self.upconv2 = nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2 = GCN_v2._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2")
            self.conv1_2 = GCN_v2._singleblock(base_n_filter * 2, base_n_filter, name="conv1_2")

            self.decoder3_high_level = GCN_v2._singleblock(base_n_filter * 4, (base_n_filter * 4),
                                                           name="dec3_high_level")
            self.decoder3_GCN = GBA(input_num=base_n_filter * 4, output_channels=base_n_filter * 4,
                                    hidden_layer=base_n_filter * 4, class_num=n_classes)
            # self.decoder3_GCN = GCN_v1._singleblock((base_n_filter * 2) * 2, base_n_filter * 4, name="dec3_GCN")

            self.upconv2_GCN = nn.ConvTranspose3d(base_n_filter * 4 * 2, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder2_GCN = GCN_v2._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2_GCN")

            self.upconv1 = nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1 = GCN_v2._block(base_n_filter * 2, base_n_filter, name="dec1")

            self.conv = nn.Conv3d(in_channels=base_n_filter, out_channels=n_classes, kernel_size=1)

        if int(self.DM_only) == 1:
            self.upconv4_loc = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size=2, stride=2)
            self.decoder4_loc = GCN_v2._block((base_n_filter * 8) * 2, base_n_filter * 8, name="dec4")
            self.upconv3_loc = nn.ConvTranspose3d(base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2)
            self.decoder3_loc = GCN_v2._block((base_n_filter * 4) * 2, base_n_filter * 4, name="dec3")
            self.upconv2_loc = nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2)
            self.decoder2_loc = GCN_v2._block((base_n_filter * 2) * 2, base_n_filter * 2, name="dec2")
            self.upconv1_loc = nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=2, stride=2)
            self.decoder1_loc = GCN_v2._block(base_n_filter * 2, base_n_filter, name="dec1")

            self.conv_loc = nn.Conv3d(in_channels=base_n_filter, out_channels=1, kernel_size=1)

    def forward(self, x, phase, network_switch, deep_supervision=False, mask=None):
        if phase != 'val_labeled':
            if phase == 'trainLabeled':
                encoder = True
                decoder_seg = True
                decoder_am = True
            if phase == 'trainLabeled_seg':
                encoder = True
                decoder_seg = True
                decoder_am = False
            if phase == 'trainLabeled_am':
                encoder = True
                decoder_seg = False
                decoder_am = True
            if phase == 'trainLabeled_am':
                encoder = True
                decoder_seg = False
                decoder_am = True

            # Set trainable parameters for Shared Encoder
            for param in self.encoder1.parameters():
                param.requires_grad = encoder
            for param in self.encoder2.parameters():
                param.requires_grad = encoder
            for param in self.encoder3.parameters():
                param.requires_grad = encoder
            for param in self.encoder4.parameters():
                param.requires_grad = encoder

            for param in self.e_pool1.parameters():
                param.requires_grad = encoder
            for param in self.e_pool2.parameters():
                param.requires_grad = encoder
            for param in self.e_pool3.parameters():
                param.requires_grad = encoder
            for param in self.e_pool4.parameters():
                param.requires_grad = encoder
            for param in self.bottleneck_seg_2.parameters():
                param.requires_grad = decoder_seg

            # MLP AM
            if int(self.AM_only) == 1:
                for param in self.am_pool1.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool2.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool3.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool4.parameters():
                    param.requires_grad = decoder_am
                for param in self.am_pool_bn.parameters():
                    param.requires_grad = decoder_am

                for param in self.upconv4_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.upconv3_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.upconv2_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.upconv1_am.parameters():
                    param.requires_grad = decoder_am

                for param in self.decoder4_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.decoder3_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.decoder2_am.parameters():
                    param.requires_grad = decoder_am
                for param in self.decoder1_am.parameters():
                    param.requires_grad = decoder_am

                for param in self.conv_am.parameters():
                    param.requires_grad = decoder_am

            if int(self.Seg_only) == 1:
                for param in self.upconv4.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv3.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv2.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv2_GCN.parameters():
                    param.requires_grad = decoder_seg
                for param in self.upconv1.parameters():
                    param.requires_grad = decoder_seg

                for param in self.decoder1.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder2.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder2_GCN.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder3.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder3_GCN.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder3_high_level.parameters():
                    param.requires_grad = decoder_seg
                for param in self.decoder4.parameters():
                    param.requires_grad = decoder_seg
                for param in self.conv.parameters():
                    param.requires_grad = decoder_seg

                for param in self.conv1_2.parameters():
                    param.requires_grad = decoder_seg
                for param in self.conv1_3.parameters():
                    param.requires_grad = decoder_seg
                for param in self.conv1_4.parameters():
                    param.requires_grad = decoder_seg

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.e_pool1(enc1))
        enc3 = self.encoder3(self.e_pool2(enc2))
        enc4 = self.encoder4(self.e_pool3(enc3))
        bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4))

        if int(self.DM_only) == 1:
            bottleneck_loc = self.bottleneck_loc(self.e_pool4(enc4))
            # bottleneck_seg = torch.cat((bottleneck_seg, bottleneck_AM), dim=1)
            # bottleneck_seg = self.decoder_joint(bottleneck_seg)
            # for param in self.bottleneck_loc.parameters():
            #     print('bottleneck_loc', param.requires_grad)
            dec4_loc = self.upconv4_loc(bottleneck_loc)
            # for param in self.upconv4.parameters():
            #     print('dec4', param.requires_grad)
            dec4_loc = torch.cat((dec4_loc, enc4), dim=1)
            dec4_loc = self.decoder4_loc(dec4_loc)
            dec3_loc = self.upconv3_loc(dec4_loc)
            dec3_loc = torch.cat((dec3_loc, enc3), dim=1)
            dec3_loc = self.decoder3_loc(dec3_loc)
            dec2_loc = self.upconv2_loc(dec3_loc)
            dec2_loc = torch.cat((dec2_loc, enc2), dim=1)
            dec2_loc = self.decoder2_loc(dec2_loc)
            dec1_loc = self.upconv1_loc(dec2_loc)
            dec1_loc = torch.cat((dec1_loc, enc1), dim=1)
            dec1_loc = self.decoder1_loc(dec1_loc)
            out_loc = self.conv_loc(dec1_loc)
        else:
            out_loc = torch.zeros_like(x)

        if int(self.AM_only) == 1:
            am_out1 = self.am_pool_bn(bottleneck_seg)
            b, c, _, w, d = am_out1.shape
            am_out1 = am_out1.reshape(b, c, w, d)
            dec4_am = self.upconv4_am(am_out1)
            dec4_am = torch.cat((dec4_am, self.am_pool4(enc4).reshape(dec4_am.shape)), dim=1)
            dec4_am = self.decoder4_am(dec4_am)
            dec3_am = self.upconv3_am(dec4_am)
            dec3_am = torch.cat((dec3_am, self.am_pool3(enc3).reshape(dec3_am.shape)), dim=1)
            dec3_am = self.decoder3_am(dec3_am)
            dec2_am = self.upconv2_am(dec3_am)
            dec2_am = torch.cat((dec2_am, self.am_pool2(enc2).reshape(dec2_am.shape)), dim=1)
            dec2_am = self.decoder2_am(dec2_am)
            dec1_am = self.upconv1_am(dec2_am)
            dec1_am = torch.cat((dec1_am, self.am_pool1(enc1).reshape(dec1_am.shape)), dim=1)
            dec1_am = self.decoder1_am(dec1_am)
            out_AM = self.conv_am(dec1_am)
        else:
            out_AM = torch.zeros_like(x)

        if int(self.Seg_only) == 1:
            # if deep_supervision:
            #     bottleneck_seg,out_cls,mask_cls = self.bottleneck_seg_2(self.e_pool4(enc4),edge_indexs,deep_supervision,mask)
            #     # for param in self.bottleneck_seg_2.parameters():
            #     #     print('bottleneck_seg_2', param.requires_grad)
            # else:
            #     # bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4), edge_indexs) # GCN
            #     bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4))  # GCN
            #     # bottleneck_seg = self.bottleneck_seg_2(self.e_pool4(enc4)) # CNN
            # # bottleneck_seg = torch.cat((bottleneck_seg, bottleneck_AM), dim=1)
            # # bottleneck_seg = self.decoder_joint(bottleneck_seg)

            dec4 = self.upconv4(bottleneck_seg)
            # for param in self.upconv4.parameters():
            #     print('dec4', param.requires_grad)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)

            bot_fuse =  self.conv1_b(bottleneck_seg)
            dec4_fuse = self.conv1_4(dec4)
            dec3_fuse = self.conv1_3(dec3)
            dec2_fuse = self.conv1_2(dec2)
            bot_fuse = F.interpolate(bot_fuse, size=dec3.shape[2:], mode="trilinear", align_corners=True)
            dec4_fuse = F.interpolate(dec4_fuse, size=dec3.shape[2:], mode="trilinear", align_corners=True)
            dec2_fuse = F.interpolate(dec2_fuse, size=dec3.shape[2:], mode="trilinear", align_corners=True)
            gcn_dec = torch.cat((bot_fuse,dec4_fuse, dec2_fuse, dec3_fuse), dim=1)
            # print(gcn_dec.shape)
            gcn_high_level = self.decoder3_high_level(gcn_dec)

            if int(self.AM_only) == 1:
                gcn_dec, fea_logit = self.decoder3_GCN(gcn_dec, F.softmax(out_AM, 1))
            else:
                gcn_dec, fea_logit = self.decoder3_GCN(gcn_dec)
                # gcn_dec = self.decoder3_GCN(gcn_dec)
                # fea_logit = torch.zeros_like(gcn_dec)
            # gcn_combine = gcn_high_level + gcn_dec
            gcn_combine = torch.cat((gcn_high_level, gcn_dec), dim=1)

            # print(gcn_dec.shape)
            dec2_GCN = self.upconv2_GCN(gcn_combine)
            # print(dec2_GCN.shape)
            dec2_GCN = self.decoder2_GCN(dec2_GCN)
            # dec2_Combine = torch.cat((dec2_GCN, dec2), dim=1)

            dec1 = self.upconv1(dec2_GCN)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            out_seg = self.conv(dec1)
            fea_logit_display = F.interpolate(fea_logit, size=out_seg.shape[2:], mode="trilinear", align_corners=True)
        else:
            out_seg = torch.zeros_like(x)
            fea_logit_display = torch.zeros_like(x)
            fea_logit = torch.zeros_like(x)

        if deep_supervision:
            return out_seg, fea_logit_display, [fea_logit], out_AM, out_loc
        else:
            if self.n_classes == 1:
                return F.sigmoid(out_seg), out_AM, out_loc
            else:
                return out_seg, fea_logit, out_AM, out_loc

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True,
                                                       track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True,
                                                       track_running_stats=False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ]
            )
        )

    @staticmethod
    def _singleblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True,
                                                       track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                ]
            )
        )

    @staticmethod
    def _conv1block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features, eps=1e-05, momentum=0.1, affine=True,
                                                       track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                ]
            )
        )

    @staticmethod
    def _2Dblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=True,
                                                       track_running_stats=False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=True,
                                                       track_running_stats=False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ]
            )
        )