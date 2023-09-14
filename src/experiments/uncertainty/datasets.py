from pathlib import Path
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.datasets import CIFAR10C, CIFAR100C, CORRUPTION_TYPES, CORRUPTION_LEVELS


DATA_PATH = Path(__file__).resolve().parents[3] / "data"


def get_dataset(
    dataset: str, 
    corruption_type: str=None,
    corruption_level: int=0,
    train: bool=True,
    **kwargs
):

    if dataset.upper()=='CIFAR-10':
        if corruption_level:
            return cifar10c(corruption_type, corruption_level, **kwargs)
        else:
            return cifar10(**kwargs, train=train)
        
    elif dataset.upper()=='CIFAR-100':
        if corruption_level:
            return cifar100c(corruption_type, corruption_level, **kwargs)
        else:
            return cifar100(**kwargs, train=train)
        
    elif dataset.upper()=='TINYIMAGENET':
        if corruption_level:
            return tinyimagenetc(corruption_type, corruption_level, **kwargs)
        else:
            return tinyimagenet(**kwargs, train=train)
    
    else:
        raise ValueError
    

def create_corrupted_dict(zero_level: bool=True):
    clevels = [0]+CORRUPTION_LEVELS if zero_level else CORRUPTION_LEVELS
    return {ctype: {
        clevel: None for clevel in clevels
    } for ctype in CORRUPTION_TYPES}


#==========CIFAR 10====================
def cifar10(
    root: str=None,
    train: bool=True,
):
    
    root = root or DATA_PATH / "CIFAR10"
    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]

    if train:
        return datasets.CIFAR10(
            root=root,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            download=True,
        ), 3, 32, 10

    else:
        return datasets.CIFAR10(
            root=root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        ), 3, 32, 10


#==========CIFAR 10 C====================
def cifar10c(
    corruption_type: str, 
    corruption_level: int, 
    root: str=None,
):
    
    root = root or str(DATA_PATH / "CIFAR10-C")
    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]

    return CIFAR10C(
        root=root,
        corruption_type=corruption_type,
        corruption_level=corruption_level,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        download=True,
    ), 3, 32, 10


#==========CIFAR 100====================
def cifar100(
    root: str=None,
    train: bool=True,
):
    
    root = root or DATA_PATH / "CIFAR100"
    mean = [0.507, 0.487, 0.441]
    std = [0.268, 0.257, 0.276]

    if train:
        return datasets.CIFAR100(
            root=root,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            download=True,
        ), 3, 32, 100

    else:
        return datasets.CIFAR100(
            root=root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        ), 3, 32, 100
    

#==========CIFAR 100 C====================
def cifar100c(
    corruption_type: str, 
    corruption_level: int, 
    root: str=None,
):
    
    root = root or str(DATA_PATH / "CIFAR100-C")
    mean = [0.507, 0.487, 0.441]
    std = [0.268, 0.257, 0.276]

    return CIFAR100C(
        root=root,
        corruption_type=corruption_type,
        corruption_level=corruption_level,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        download=True,
    ), 3, 32, 100