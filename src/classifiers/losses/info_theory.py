import math

import torch
from torch import Tensor, nn

__all__ = [
    "ForwardCrossEntropy", 
    "SymmetricCrossEntropy", 
    "SymmetricKLDivergence"
]


class InformationTheoryLoss(nn.Module):

    def __init__(self, log_0: float=-7.):
        super().__init__()
        self.log_0 = log_0

    def forward(self) -> Tensor:
        raise NotImplementedError

    def log_mod(self, x: Tensor):
        return torch.log(x + (x==0)*math.exp(self.log_0))

    def entropy(self, preds:Tensor):
        return -torch.sum(preds*self.log_mod(preds), axis=-1)

    def forward_cross_entropy(self, preds: Tensor, labels: Tensor):
        return -torch.sum(labels*self.log_mod(preds), axis=-1)

    def reverse_cross_entropy(self, preds: Tensor, labels: Tensor):
        return -torch.sum(preds*self.log_mod(labels), axis=-1)


class ForwardCrossEntropy(InformationTheoryLoss):

    def __init__(self, log_0: float=-7.):
        super().__init__(log_0)

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        return self.forward_cross_entropy(preds, labels)


class SymmetricCrossEntropy(InformationTheoryLoss):

    def __init__(self, log_0: float=-7.):
        super().__init__(log_0)

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        fce = self.forward_cross_entropy(preds, labels)
        rce = self.reverse_cross_entropy(preds, labels)
        return fce + rce


class SymmetricKLDivergence(InformationTheoryLoss):

    def __init__(self, log_0: float=-7.):
        super().__init__(log_0)

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        ent = self.entropy(preds)
        fce = self.forward_cross_entropy(preds, labels)
        rce = self.reverse_cross_entropy(preds, labels)
        return fce + rce - ent