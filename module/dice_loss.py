import torch
import torch.nn as nn
from torch.autograd import Function, Variable


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceCoefficientLF(nn.Module):

    def __init__(self, device):
        super(DiceCoefficientLF, self).__init__()
        self.device = device

    def forward(self, y_pred,y_true):
        _smooth = torch.tensor([0.0001]).to(self.device)
        return 1.0 - (2.0 * torch.sum(y_true * y_pred)) /(torch.sum(y_true) + torch.sum(y_pred) + _smooth)


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

class DiceCoefficientLF_rec(nn.Module):

    def __init__(self, device):
        super(DiceCoefficientLF_rec, self).__init__()
        self.device = device

    def forward(self, y_true, y_pred):
        _smooth = torch.tensor([0.0001]).to(self.device)
        return (2.0 * torch.sum(y_true * y_pred)) /(torch.sum(y_true) + torch.sum(y_pred) + _smooth)


class MSELF(nn.Module):

    def __init__(self, device):
        super(MSELF, self).__init__()
        self.device = device

    def forward(self, y_true, y_pred):
        _smooth = torch.tensor([0.0001]).to(self.device)
        print(torch.max(y_true))
        print(torch.min(y_true))

        print(torch.max(y_pred))
        print(torch.min(y_pred))
        return torch.sum(y_pred - y_true)
