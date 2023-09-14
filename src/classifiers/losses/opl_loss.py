"""
https://github.com/kahnchana/opl/blob/master/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .info_theory import ForwardCrossEntropy


class OrthogonalProjectionLoss(nn.Module):
    
    def __init__(self, no_norm=False, use_attention=False, gamma=2, opl_ratio=0.1):
        super(OrthogonalProjectionLoss, self).__init__()
        self.weights_dict = None
        self.no_norm = no_norm
        self.gamma = gamma
        self.opl_ratio = opl_ratio
        self.use_attention = use_attention
        self.base_loss = ForwardCrossEntropy()


    def forward(self, features, preds, labels):

        base_loss = self.base_loss(preds, labels)

        labels = labels.argmax(axis=1)

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if self.use_attention:
            features_weights = torch.matmul(features, features.T)
            features_weights = F.softmax(features_weights, dim=1)
            features = torch.matmul(features_weights, features)

        #  features are normalized
        if not self.no_norm:
            features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)

        return base_loss + self.opl_ratio*loss