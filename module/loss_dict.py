from __future__ import print_function, division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
import torchvision.models as models


class TotalVariationLoss(torch.nn.Module):
    def __init__(self, do_3d=False):
        super().__init__()
        self.do_3d = do_3d

    def forward(self, x):
        if self.do_3d:
            loss = torch.sum(torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])) + \
                   torch.sum(torch.abs(x[:, :, :, :-1, :] - x[:, :, :, 1:, :])) + \
                   torch.sum(torch.abs(x[:, :, :-1, :, :] - x[:, :, 1:, :, :]))
        else:
            loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                   torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return loss


class TransferLoss(nn.Module):
    def __init__(self, con=1.0, sty=1e6, fea_idx={'con': [1], 'sty': [1]}, style_loss='gram', **kwargs):
        super().__init__()
        self.w_con = con
        self.w_sty = sty
        assert style_loss in ['gram', 'norm']
        self.style_loss = style_loss
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = VGG(self.vgg16.features[:23]).cuda().eval()
        self.fea_idx = fea_idx

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def get_features(self, x):
        # return_feas = []
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feas = self.vgg16(x)
        # for i in self.fea_idx:
        #     return_feas.append(feas[i])
        return feas

    def eval(self, pred, cont, style):
        pred_feas = self.get_features(pred)
        cont_feas = self.get_features(cont)
        loss = 0
        ld = {}
        num = len(pred_feas)

        loss_con = 0
        for i in self.fea_idx['con']:
            loss_con += F.mse_loss(pred_feas[i], cont_feas[i]) * self.w_con
        loss += loss_con
        ld['con'] = loss_con.item()

        loss_sty = 0
        style_feas = self.get_features(style)
        if self.style_loss == 'gram':
            style_grams = [self.gram_matrix(style_feas[i]).detach() for i in self.fea_idx['sty']]
            pred_grams = [self.gram_matrix(pred_feas[i]) for i in self.fea_idx['sty']]
            for i in range(len(style_grams)):
                loss_sty += F.mse_loss(pred_grams[i], style_grams[i]) * self.w_sty
        elif self.style_loss == 'norm':
            style_means = [torch.mean(style_feas[i]).detach() for i in self.fea_idx['sty']]
            style_stds = [torch.std(style_feas[i]).detach() for i in self.fea_idx['sty']]
            pred_means = [torch.mean(pred_feas[i]) for i in self.fea_idx['sty']]
            pred_stds = [torch.std(pred_feas[i]) for i in self.fea_idx['sty']]
            for i in range(len(style_means)):
                loss_sty += (F.mse_loss(pred_means[i], style_means[i]) + F.mse_loss(pred_stds[i],
                                                                                    style_stds[i])) * self.w_sty

        loss += loss_sty
        ld['sty'] = loss_sty.item()

        return loss, ld


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",  # 64, 512, 512
            '8': "relu2_2",  # 128, 256, 256
            '15': "relu3_3",  # 256, 128, 128
            '22': "relu4_3"  # 512, 64, 64
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


class SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, do_3d=False):
        super().__init__()
        if do_3d:
            self.ssim = SSIM_3D(window_size, size_average)
        else:
            self.ssim = SSIM(window_size, size_average)

    def forward(self, img1, img2, mask=None):
        return 1 - self.ssim(img1, img2, mask)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True, mask=None):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map = (sigma12 + C2/2) / (sigma1_sq * sigma2_sq + C2/2)
        if mask is not None:
            mask = F.max_pool2d(1 - mask.float(), kernel_size=window_size, stride=1, padding=window_size // 2).bool()
            ssim_map[mask] = 0

            if size_average:
                return ssim_map[~mask].mean()
            else:
                return ssim_map[~mask]

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, img1, img2, mask=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average, mask)


class SSIM_3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5)
        _3D_window = _1D_window[None, None, :] * _1D_window[None, :, None] * _1D_window[:, None, None]
        _3D_window = _3D_window.float().unsqueeze(0).unsqueeze(0)
        window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True, mask=None):
        img1 = F.conv2d(img1, Gaussian_Kernel, padding=1, stride=1)
        img2 = F.conv2d(img2, Gaussian_Kernel, padding=1, stride=1)
        mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if mask is not None:
            mask = F.max_pool3d(1 - mask.float(), kernel_size=window_size, stride=1, padding=window_size // 2).bool()
            ssim_map[mask] = 0

            if size_average:
                return ssim_map[~mask].mean()
            else:
                return ssim_map[~mask]

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, img1, img2, mask=None):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average, mask)


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
                pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha
        distance = distance.to(pred.device)

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


#######################################################
# 0. Main loss functions
#######################################################

class CustomKLLoss(nn.Module):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''

    def __init__(self, average=True):
        super(CustomKLLoss, self).__init__()
        self.average = average

    def forward(self, mean, logvar):
        std = torch.exp(logvar / 2)
        if self.average:
            return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(
                torch.log(torch.mul(std, std))) - 1
        else:
            bs = mean.size(0)
            return torch.sum(torch.mul(mean, mean) + torch.mul(std, std) - torch.log(torch.mul(std, std)) - 1) / bs


class HuberLoss(nn.Module):
    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        residual = (pred - target).abs()
        loss = 0
        if (residual <= self.delta).max() == True:
            loss += 0.5 * (residual[residual <= self.delta]).pow(2).mean()
        if (residual > self.delta).max() == True:
            loss += (residual[residual > self.delta]).mean() * self.delta - 0.5 * self.delta ** 2
        return loss


# class FocalLoss(nn.Module):
#     """
#     This class implements the segmentation focal loss.
#     https://arxiv.org/abs/1708.02002
#     """
#
#     def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
#         """
#         Constructor method
#         :param alpha: (float) Alpha constant
#         :param gamma: (float) Gamma constant (see paper)
#         """
#         # Call super constructor
#         super(FocalLoss, self).__init__()
#         # Save parameters
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def __repr__(self):
#         """
#         Get representation of the loss module
#         :return: (str) String including information
#         """
#         return "{}, alpha={}, gamma={}".format(self.__class__.__name__, self.alpha, self.gamma)
#
#     def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass computes the binary cross entropy loss of segmentation masks
#         :param prediction: (torch.Tensor) Prediction probability
#         :param label: (torch.Tensor) Label one-hot encoded
#         :return: (torch.Tensor) Loss value
#         """
#         # Calc binary cross entropy loss as normal
#         binary_cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-12))
#                                       + (1.0 - label) * torch.log((1.0 - prediction).clamp(min=1e-12)))
#         # Calc focal loss factor based on the label and the prediction
#         focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
#         # Calc final focal loss
#         loss = ((1.0 - focal_factor) ** self.gamma *
#                 binary_cross_entropy_loss * self.alpha).sum(dim=1).mean()
#         return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class JaccardLoss(nn.Module):
    """Jaccard loss.
    """

    # binary case

    def __init__(self, size_average=True, reduce=True, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce

    def jaccard_loss(self, pred, target):
        loss = 0.
        # for each sample in the batch
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - ((intersection + self.smooth) /
                         (iflat.sum() + tflat.sum() - intersection + self.smooth))
            # print('loss:',intersection, iflat.sum(), tflat.sum())

        # size_average=True for the jaccard loss
        return loss / float(pred.size()[0])

    def jaccard_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((intersection + self.smooth) /
                    (iflat.sum() + tflat.sum() - intersection + self.smooth))
        # print('loss:',intersection, iflat.sum(), tflat.sum())
        return loss

    def forward(self, pred, target):
        # _assert_no_grad(target)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))
        if self.reduce:
            loss = self.jaccard_loss(pred, target)
        else:
            loss = self.jaccard_loss_batch(pred, target)
        return loss


class DiceLoss(nn.Module):
    """DICE loss.
    """

    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    def __init__(self, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        shape = pred.shape
        iflat = pred.view(shape[0], shape[1], -1)
        tflat = target.view((shape[0], shape[1], -1))
        intersection = (iflat * tflat).sum(-1)
        if self.power == 1:
            loss = 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum(-1) + tflat.sum(-1) + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) /
                        ((iflat ** self.power).sum(-1) + (tflat ** self.power).sum(-1) + self.smooth))

        # size_average=True for the dice loss
        return loss.mean()

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.power == 1:
            loss = 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) /
                        ((iflat ** self.power).sum() + (tflat ** self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target):
        # _assert_no_grad(target)
        # pred shape: (bs, c, ...)
        # target shape: (bs, c, ...)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


# class DiceLoss(nn.Module):
#     """DICE loss.
#     """
#     # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

#     def __init__(self, reduce=True, smooth=100.0, power=1):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.reduce = reduce
#         self.power = power

#     def dice_loss(self, pred, target):
#         loss = 0.

#         for index in range(pred.size()[0]):
#             iflat = pred[index].view(-1)
#             tflat = target[index].view(-1)
#             intersection = (iflat * tflat).sum()
#             if self.power == 1:
#                 loss += 1 - ((2. * intersection + self.smooth) /
#                         ( iflat.sum() + tflat.sum() + self.smooth))
#             else:
#                 loss += 1 - ((2. * intersection + self.smooth) /
#                         ( (iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))

#         # size_average=True for the dice loss
#         return loss / float(pred.size()[0])

#     def dice_loss_batch(self, pred, target):
#         iflat = pred.view(-1)
#         tflat = target.view(-1)
#         intersection = (iflat * tflat).sum()

#         if self.power==1:
#             loss = 1 - ((2. * intersection + self.smooth) /
#                    (iflat.sum() + tflat.sum() + self.smooth))
#         else:
#             loss = 1 - ((2. * intersection + self.smooth) /
#                    ( (iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))
#         return loss

#     def forward(self, pred, target):
#         #_assert_no_grad(target)
#         # pred shape: (bs, c, ...)
#         # target shape: (bs, c, ...)
#         if not (target.size() == pred.size()):
#             raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))

#         if self.reduce:
#             loss = self.dice_loss(pred, target)
#         else:
#             loss = self.dice_loss_batch(pred, target)
#         return loss

class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        else:
            return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight=None):
        # _assert_no_grad(target)
        return self.weighted_mse_loss(pred, target, weight)


class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """

    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight=None):
        # _assert_no_grad(target)
        return F.binary_cross_entropy(pred, target, weight)


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        # Different from, F.binary_cross_entropy, the "weight" parameter
        # in F.cross_entropy is a manual rescaling weight given to each
        # class. Therefore we need to multiply the weight mask after the
        # loss calculation.
        loss = F.cross_entropy(pred, target, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


class WeightedNLL(nn.Module):
    """Mask weighted nll loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight=None):
        loss = F.nll_loss(pred, target, weight)
        return loss

def cos_sim(pred, target):
    return F.cosine_similarity(pred, target, dim=-1).mean()

def l2_loss(values):
    return F.mse_loss(values, torch.zeros_like(values))

class cosine_similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        cos_sim_1 = cos_sim(pred.view(pred.shape[0], -1), target.view(target.shape[0], -1))
        loss = l2_loss(cos_sim_1 - torch.ones_like(cos_sim_1))
        return loss

Gaussian_Kernel = torch.tensor([[1,2,1],
                   [2,4,2],
                   [1,2,1]],dtype=torch.float32,requires_grad=False).cuda()
Gaussian_Kernel = torch.nn.Parameter(Gaussian_Kernel.view(1,1,3,3)) / 16

class Guassian_MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # print('before',target)
        pred_gaussian = F.conv2d(pred, Gaussian_Kernel, padding=1, stride=1)
        target_gaussian = F.conv2d(target, Gaussian_Kernel, padding=1, stride=1)
        # print('after', target_gaussian)
        # print('pred',pred)
        # print('gau',pred_gaussian)
        loss = nn.MSELoss()(pred_gaussian,target_gaussian)
        # loss = nn.MSELoss()(pred, target)
        return loss

#######################################################
# 1. Regularization
#######################################################

class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred):
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = (1.0 / diff).mean()
        return self.alpha * loss
