from pathlib import Path
import torchvision.transforms as transforms
import torchvision.datasets as datasets

DATA_PATH = Path(__file__).resolve().parents[3] / "data"


def get_dataset(dataset, **kwargs):
    if dataset.upper()=='MNIST':
        return mnist(**kwargs)
    elif dataset.upper()=='CIFAR-10':
        return cifar10(**kwargs)
    elif dataset.upper()=='CIFAR-100':
        return cifar100(**kwargs)
    elif dataset.upper()=='SVHN':
        return svhn(**kwargs)
    else:
        raise ValueError


#==========MNIST====================
def mnist(
    root: str=None,
    train: bool=True,
):
    
    root = root or DATA_PATH / "MNIST"
    mean=[0.1307,]
    std=[0.3081,]

    if train:
        return datasets.MNIST(
            root=root,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Pad(2),
            ]),
            download=True,
        ), 1, 32, 10

    else:
        return datasets.MNIST(
            root=root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Pad(2),
            ])
        ), 1, 32, 10


#==========CIFAR 10====================
def cifar10(
    root: str=None,
    train: bool=True,
):
    
    root = root or DATA_PATH / "CIFAR10"
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.2435, 0.2616]

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


#==========CIFAR 100====================
def cifar100(
    root: str=None,
    train: bool=True,
):
    
    root = root or DATA_PATH / "CIFAR100"
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    if train:
        return datasets.CIFAR100(
            root=root,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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
    

#==========SVHN====================
def svhn(
    root: str=None,
    train: bool=True,
):
    
    root = root or DATA_PATH / "SVHN"
    mean = [0.4377, 0.4437, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    if train:
        return datasets.SVHN(
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
        return datasets.SVHN(
            root=root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        ), 3, 32, 10
    