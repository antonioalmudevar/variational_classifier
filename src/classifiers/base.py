from typing import OrderedDict

from torch import nn, Tensor

from .losses import *

class BaseClassifier(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        class_loss_fn: str='fce',
        log_0: float=-7.,
        n_samples: int=1
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.n_classes = n_classes
        self.embed_dim = encoder.size_code
        self.n_samples = n_samples

        self._get_class_head(class_loss_fn)
        self._get_class_loss_fn(class_loss_fn, log_0)
        
    #==========Forward==========
    def forward(self) -> Tensor:
        raise NotImplementedError
    
    #==========Embeds to Logits==========
    def _get_class_head(self, class_loss_fn: str):
        if class_loss_fn.upper()=='RBF':
            self.class_head = RBFLogits(self.embed_dim, self.n_classes)
        elif class_loss_fn.upper()=='SPHEREFACE':
            self.class_head = AngleLinear(self.embed_dim, self.n_classes)
        else:
            self.class_head = nn.Linear(self.embed_dim, self.n_classes)

    def get_logits(self, embeds: Tensor):
        return self.class_head(embeds)

    #==========Logits to Predictions==========
    def get_preds(self, logits: Tensor):
        if isinstance(self.class_loss_fn, AngleLoss):
            cos_theta, _ = logits
            return cos_theta.softmax(dim=-1)
        else:
            return logits.softmax(dim=-1)
        
    #==========Losses==========
    def _get_class_loss_fn(self, class_loss_fn: str, log_0: float):
        if class_loss_fn.upper() in ['FCE', 'RBF']:
            self.class_loss_fn = ForwardCrossEntropy(log_0)
        elif class_loss_fn.upper()=='ANCHOR':
            self.class_loss_fn = AnchorLoss()
        elif class_loss_fn.upper()=='FOCAL':
            self.class_loss_fn = FocalLoss()
        elif class_loss_fn.upper()=='OPL':
            self.class_loss_fn = OrthogonalProjectionLoss()
        elif class_loss_fn.upper()=='SPHEREFACE':
            self.class_loss_fn = AngleLoss()
        else:
            raise ValueError
        
    def class_loss(
        self, 
        labels: Tensor,
        embeds: Tensor=None,
        logits: Tensor=None,
        preds: Tensor=None,
    ):
        if isinstance(self.class_loss_fn, OrthogonalProjectionLoss):
            return self.class_loss_fn(embeds, preds, labels)
        elif isinstance(self.class_loss_fn, (FocalLoss, AngleLoss)):
            return self.class_loss_fn(logits, labels)
        else:
            return self.class_loss_fn(preds, labels)
        
    def calc_loss(self):
        raise NotImplementedError
    
    #==========Load model==========
    def load_state_dict(
        self, 
        state_dict: OrderedDict[str, Tensor], 
        strict: bool = True,
        copy_head_layer: bool=True
    ):
        if copy_head_layer:
            return super().load_state_dict(state_dict, strict)
        else:
            return self.encoder.load_state_dict(state_dict, strict)
    
    def n_samples(self):
        return self.n_samples
    


class BaseClassifierParallel(nn.DataParallel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load_state_dict(
        self, 
        state_dict: OrderedDict[str, Tensor], 
        strict: bool = True,
        copy_head_layer: bool=True
    ):
        return self.module.load_state_dict(state_dict, strict, copy_head_layer)

    def state_dict(self):
        return self.module.state_dict()

    def calc_loss(self, *args, **kwargs) -> Tensor:
        return self.module.calc_loss(*args, **kwargs)
    
    def class_loss(self, *args, **kwargs) -> Tensor:
        return self.module.class_loss(*args, **kwargs)
    
    def kl_loss(self, *args, **kwargs) -> Tensor:
        return self.module.kl_loss(*args, **kwargs)
    
    def n_samples(self):
        return self.module.n_samples