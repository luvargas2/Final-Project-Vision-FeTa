# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utilities.utils import one_hot
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
#from libs.utilities.evaluation_metrics import haussdorff_distance

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        n, ch, x, y, z = inputs.size()

        logpt = -self.criterion(inputs, targets.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class tversky_loss(nn.Module):
    """
        Calculates the Tversky loss of the Foreground categories.
        if alpha == 1 --> Dice score
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
    """
    def __init__(self, alpha, eps=1e-5):
        super(tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta = 2 - alpha
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs.shape[1]: predicted categories
        targets = one_hot(targets, inputs.shape[1])
        inputs = F.softmax(inputs, dim=1)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims) * self.alpha
        fns = torch.sum((1 - inputs) * targets, dims) * self.beta
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        loss = torch.mean(loss, dim=0)
        return 1 - (loss[1:]).mean()


class segmentation_loss(nn.Module):
    def __init__(self, alpha):
        super(segmentation_loss, self).__init__()
        self.dice = tversky_loss(alpha=alpha, eps=1e-5)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets.contiguous())
        ce = self.ce(inputs, targets)
        return dice + ce

class segmentation_loss3D(nn.Module):
    def __init__(self, alpha,percentile):
        super(segmentation_loss3D, self).__init__()
        self.dice = tversky_loss(alpha=alpha, eps=1e-5)
        self.ce = nn.CrossEntropyLoss()
        self.haussdorf = HausdorffDTLoss(percentile=percentile)

    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets.contiguous())
        ce = self.ce(inputs, targets)
        haussdorf = self.haussdorf(inputs, targets)
        return dice + ce + haussdorf


class Dice_metric(nn.Module):
    def __init__(self, eps=1e-5):
        super(Dice_metric, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):
        categories = inputs.shape[1]
        targets = targets.contiguous()
        targets = one_hot(targets, categories)
        if logits:
            inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        inputs = one_hot(inputs, categories)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims)
        fns = torch.sum((1 - inputs) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)

# class HaussdorfLoss(nn.Module):
#     def __init__(self, percentile=95):
#         super(HaussdorfLoss, self).__init__()
#         self.percentile = percentile

#     def forward(self, inputs, targets):
#         # inputs.shape[1]: predicted categories
#         targets = one_hot(targets, inputs.shape[1]).cpu()
#         inputs = F.softmax(inputs, dim=1).cpu()
#         clases=[1,2,3,4,5,6,7]
#         targets = targets.detach().numpy()
#         inputs = inputs.detach().numpy()
#         print('input',  inputs.shape, 'target', targets.shape)
#         print('input',type(inputs), 'target', type(targets))
#         hd_loss = haussdorff_distance(inputs, targets,clases,percentile=95)

#         return hd_loss

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, percentile=95, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        self.percentile = percentile

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
        categories = pred.shape[1]
        target = target.contiguous()
        target = one_hot(target, categories)

        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)
        with torch.no_grad():
            pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
            target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha
        dt_field = pred_error.cuda() * distance.cuda()
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

