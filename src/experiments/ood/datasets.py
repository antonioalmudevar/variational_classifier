from pathlib import Path
import torchvision.transforms as transforms
import torchvision.datasets as datasets

DATA_PATH = Path(__file__).resolve().parents[3] / "data"


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
            split='train',
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
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            download=True,
        ), 3, 32, 10
    