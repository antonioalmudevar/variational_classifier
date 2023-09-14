from typing import Union, Tuple, List, TypedDict

from .encoders import get_encoder
from .base import BaseClassifierParallel
from .deterministic import DeterministicClassifier
from .variational import VariationalClassifier
from .variational_mean import VariationalMeanClassifier


def get_class_arch(
    cfg_encoder: TypedDict, 
    n_classes: int,
    ch_in: int,
    size_in: Union[int, Tuple[int], List[int]],
    class_type: str,
    **kwargs
):
    encoder = get_encoder(ch_in=ch_in, size_in=size_in, **cfg_encoder)

    if class_type.upper()=='DETERMINISTIC':
        classifier = DeterministicClassifier
    elif class_type.upper()=='VARIATIONAL':
        classifier = VariationalClassifier
    elif class_type.upper()=='VARIATIONAL_MEAN':
        classifier = VariationalMeanClassifier

    return classifier(encoder=encoder, n_classes=n_classes, **kwargs)