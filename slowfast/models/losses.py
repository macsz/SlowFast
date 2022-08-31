#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class RMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.reduction = reduction

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, reduction="mean"):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        #BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        loss = torch.sqrt(self.mse(inputs,targets))
        targets = targets.type(torch.long)
        #at = self.alpha.gather(0, targets.view(-1))
        pt = torch.exp(loss)
        F_loss = (1-pt)**2 * loss
        return F_loss.mean()


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "mse": nn.MSELoss,
    "focal": WeightedFocalLoss,
    "rmse": RMSELoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
