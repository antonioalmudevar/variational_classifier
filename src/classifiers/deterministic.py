from torch import Tensor

from .base import BaseClassifier


class DeterministicClassifier(BaseClassifier):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    #==========Forward==========
    def _encode(self, x: Tensor, n_samples: int) -> Tensor:
        embeds = self.encoder(x, n_samples=n_samples)
        return embeds

    def forward(self, x: Tensor, n_samples: int=None) -> Tensor:
        n_samples = 1 if n_samples is None else n_samples
        embeds = self._encode(x, n_samples=n_samples)
        logits = self.get_logits(embeds)
        preds = self.get_preds(logits)
        if n_samples>1:
            preds = preds.reshape(-1, n_samples, *preds.shape[1:]).mean(dim=1)
        return {'embeds': embeds, 'logits': logits, 'preds': preds}
    
    #==========Losses==========
    def calc_loss(self, labels: Tensor, embeds: Tensor, logits: Tensor, preds: Tensor):
        return self.class_loss(labels, embeds, logits, preds)