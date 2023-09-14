from typing import Union, Tuple, List

from . import cifar_resnet
from . import resnet

def get_encoder(
    arch, 
    ch_in: int,
    size_in: Union[int, Tuple[int], List[int]],
    **kwargs
):
    if arch in cifar_resnet.MODELS:
        return cifar_resnet.select_cifar_resnet(arch=arch, ch_in=ch_in, **kwargs)
    
    elif arch in resnet.MODELS:
        return resnet.ResNet(model_size=arch, **kwargs)
    
    else:
        raise ValueError