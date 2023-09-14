from typing import Union, Tuple, List

import torch

from src.classifiers import get_class_arch
from src.helpers.training import read_config, save_config, get_models_list

__all__ = [
    "get_classifier",
    "read_configs_classifier",
    "load_epoch_classifier", 
    "load_last_epoch_classifier",
]


def get_classifier(
    cfg_encoder,
    cfg_classifier,
    n_classes: int,
    ch_in: int,
    size_in: Union[int, Tuple[int, int], List[int]],
    device: torch.device,
):
    return get_class_arch(
        cfg_encoder=cfg_encoder,
        n_classes=n_classes,
        ch_in=ch_in,
        size_in=size_in,
        **cfg_classifier,
    ).to(device=device, dtype=torch.float)


def read_configs_classifier(
    path,
    model_dir, 
    config_data, 
    config_encoder,
    config_classifier, 
    config_training,
    save: bool=False,
):
    cfg_data = read_config(path/"data"/config_data)
    cfg_encoder = read_config(path/"encoder"/config_encoder)
    cfg_class = read_config(path/"classifier"/config_classifier)
    cfg_training = read_config(path/"training"/config_training)

    if save:
        save_config(cfg_data, model_dir/"config_data")
        save_config(cfg_encoder, model_dir/"config_encoder")
        save_config(cfg_class, model_dir/"config_classifier")
        save_config(cfg_training, model_dir/"config_training")

    return cfg_data, cfg_encoder, cfg_class, cfg_training
   

def load_last_epoch_classifier(
    model_dir: str,
    classifier: Union[torch.nn.Module, torch.nn.DataParallel],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    restart: bool=False
):
    previous_models = get_models_list(model_dir, 'epoch_')
    if len(previous_models)>0 and not restart:
        checkpoint = torch.load(model_dir/previous_models[-1])
        classifier.load_state_dict(checkpoint['classifier'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch']
    else:
        return 0


def load_epoch_classifier(
    epoch: int,
    device: torch.device,
    model_dir: str,
    classifier: Union[torch.nn.Module, torch.nn.DataParallel]=None,
    optimizer: torch.optim.Optimizer=None,
    scheduler: torch.optim.lr_scheduler._LRScheduler=None,
):
    epoch_path = model_dir/("epoch_"+str(epoch)+".pt")
    if classifier is not None:
        classifier.load_state_dict(
            torch.load(epoch_path, map_location=device)['classifier']
        )
    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(epoch_path, map_location=device)['optimizer']
        )
    if scheduler is not None:
        scheduler.load_state_dict(
            torch.load(epoch_path, map_location=device)['scheduler']
        )


def save_epoch_classifier(
    epoch: int,
    model_dir: str,
    classifier: Union[torch.nn.Module, torch.nn.DataParallel],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    save: bool=True,
):
    if save:
        epoch_path = model_dir/("epoch_"+str(epoch)+".pt")
        classifier_state_dict = classifier.module.state_dict() if \
            isinstance(classifier, torch.nn.DataParallel) else classifier.state_dict()
        checkpoint = {
            'epoch':        epoch,
            'classifier':   classifier_state_dict,
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
        }
        torch.save(checkpoint, epoch_path)