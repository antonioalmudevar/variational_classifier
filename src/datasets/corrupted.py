from typing import Callable, Optional, Tuple, Any

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

__all__ = [
    "CORRUPTION_TYPES",
    "CORRUPTION_LEVELS",
    "CIFAR10C",
    "CIFAR100C",
]


CORRUPTION_TYPES = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "frost",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate"
]

CORRUPTION_LEVELS = [1,2,3,4,5]


#==========CIFAR-10====================
class CIFAR10C(VisionDataset):

    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"
    subroot = "CIFAR-10-C"

    def __init__(
        self, 
        root: str,
        corruption_type: str, 
        corruption_level: int, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        
        assert corruption_type in CORRUPTION_TYPES, "corruption_type is not available"
        assert corruption_level in CORRUPTION_LEVELS, "corruption_level is not available"
        
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        idx_range = range((corruption_level-1)*10000, corruption_level*10000)
        self.data = np.load(
            root+"/"+self.subroot+"/"+corruption_type+".npy",
            allow_pickle=True
        )[idx_range]
        self.targets = list(np.load(
            root+"/"+self.subroot+"/labels.npy",
            allow_pickle=True
        )[idx_range].astype(np.int64))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def download(self) -> None:
        print("Files already downloaded and verified")
        return download_and_extract_archive(
            self.url, self.root, filename=self.filename
        )


#==========CIFAR-100====================
class CIFAR100C(CIFAR10C):

    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
    subroot = "CIFAR-100-C"


def _add_channels(img):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img