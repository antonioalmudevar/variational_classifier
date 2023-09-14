from typing import Tuple

from scipy.stats import ortho_group
import torch
from torch import nn, Tensor

from .base import BaseClassifier
from ..utils import jl_transform

class VariationalClassifier(BaseClassifier):

    def __init__(
        self,
        mean_orth: bool=False,
        mean_prior: float=None,
        var_equi: bool=False,
        var_prior: float=1.,
        beta_class: float=1.,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._init_mean_class(mean_orth, mean_prior)
        self._init_var_class(var_equi, var_prior)
        self.beta_class = beta_class

        self.fc_mu = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh()
        )

    #==========Get mean and variance of each class==========
    def _init_mean_class(self, mean_orth: bool, mean_prior: float):
        if mean_orth:
            if self.embed_dim >= self.n_classes:
                mean_class = ortho_group.rvs(dim=self.embed_dim)[:self.n_classes]
            else:
                mean_class = ortho_group.rvs(dim=self.n_classes)
                mean_class = jl_transform(mean_class, self.embed_dim)
            self.mean_class = torch.Tensor(mean_class)
            if mean_prior:
                self.mean_prior = torch.Tensor([mean_prior])
            else:
                self.mean_prior = nn.Parameter(0.1*torch.randn(1))
        else:
            self.mean_class = nn.Parameter(
                0.1*torch.randn((self.n_classes, self.embed_dim))
            )
            self.mean_prior = torch.Tensor([1])

    def _init_var_class(self, var_equi: bool, var_prior: float):
        if var_equi:
            self.logvar_class = torch.log(torch.ones((self.n_classes, self.embed_dim)))
            if var_prior:
                self.logvar_prior = torch.log(torch.Tensor([var_prior]))
            else:
                self.logvar_prior = nn.Parameter(torch.log(0.1*torch.randn(1)+1))
        else:
            self.logvar_class = nn.Parameter(
                torch.log(0.1*torch.randn((self.n_classes, self.embed_dim))+1)
            )
            self.logvar_prior = torch.Tensor([1])
        
    #==========Forward==========
    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar

    def _reparametrize(self, mu: Tensor, logvar: Tensor, n_samples: int=None) -> Tensor:

        if n_samples==0:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn((tuple([n_samples])+std.shape)).to(mu.device)
            z = eps*std + mu
            z = z.swapaxes(0,1).reshape(-1, z.shape[2])
            return z

    def forward(self, x: Tensor, n_samples: int=None) -> Tensor:
        n_samples = self.n_samples if n_samples is None else n_samples
        mu, log_var = self._encode(x)
        embeds = self._reparametrize(mu, log_var, n_samples)
        logits = self.get_logits(embeds)
        preds = self.get_preds(logits)
        if n_samples>1:
            preds = preds.reshape(-1, n_samples, *preds.shape[1:]).mean(dim=1)
        return {'mu': mu, 'log_var': log_var, 'logits': logits, 'preds': preds}

    #==========Losses==========
    def kl_loss(self, mu: Tensor, log_var: Tensor, labels: Tensor) -> Tensor:
        mu_y = self.mean_prior*torch.matmul(labels, self.mean_class)
        var_y = (torch.exp(self.logvar_prior))*torch.matmul(
            labels, torch.exp(self.logvar_class)
        )
        return -0.5 * torch.sum(
            1 + \
            log_var - torch.log(var_y) - \
            (mu - mu_y)**2 / var_y - \
            log_var.exp() / var_y, 
            axis=1
        )

    def calc_loss(
        self,
        labels: Tensor,
        mu: Tensor,
        log_var: Tensor,
        logits: Tensor,
        preds: Tensor,
    ):
        
        kl_loss = self.kl_loss(mu, log_var, labels)
        class_loss = self.class_loss(labels=labels, preds=preds)
        total_loss = self.beta_class*class_loss + kl_loss

        class_norm = torch.sum(class_loss)
        total_norm = torch.sum(total_loss)

        return total_loss*class_norm/total_norm
    
    #==========Load & Save model==========
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if not isinstance(self.mean_prior, nn.Parameter):
            state_dict['mean_prior'] = self.mean_prior
        if not isinstance(self.mean_class, nn.Parameter):
            state_dict['mean_class'] = self.mean_class
        if not isinstance(self.logvar_prior, nn.Parameter):
            state_dict['logvar_prior'] = self.logvar_prior
        if not isinstance(self.logvar_class, nn.Parameter):
            state_dict['logvar_class'] = self.logvar_class
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool=False):
        if not isinstance(self.mean_prior, nn.Parameter):
            self.mean_prior = state_dict['mean_prior']
            del state_dict['mean_prior']
        if not isinstance(self.mean_class, nn.Parameter):
            self.mean_class = state_dict['mean_class']
            del state_dict['mean_class']
        if not isinstance(self.logvar_prior, nn.Parameter):
            self.logvar_prior = state_dict['logvar_prior']
            del state_dict['logvar_prior']
        if not isinstance(self.logvar_class, nn.Parameter):
            self.logvar_class = state_dict['logvar_class']
            del state_dict['logvar_class']
        super().load_state_dict(state_dict, strict)

    #==========Device==========
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mean_prior = self.mean_prior.to(*args, **kwargs)
        self.mean_class = self.mean_class.to(*args, **kwargs)
        self.logvar_prior = self.logvar_prior.to(*args, **kwargs)
        self.logvar_class = self.logvar_class.to(*args, **kwargs)
        return self
