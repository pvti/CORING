from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *

class DataLoaderCIFAR10:
    def __init__(self):
        self.image_size = 32
        self.transforms = {
            "train": Compose([
                RandomCrop(self.image_size, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
            ]),
            "test": ToTensor(),
        }
        self.dataset = {}
        self.dataloader = {}
        for split in ["train", "test"]:
            self.dataset[split] = CIFAR10(root="data/cifar10",
                                          train=(split == "train"),
                                          download=True,
                                          transform=self.transforms[split],
                                          )

        for split in ['train', 'test']:
            self.dataloader[split] = DataLoader(self.dataset[split],
                                                batch_size=512,
                                                shuffle=(split == 'train'),
                                                num_workers=0,
                                                pin_memory=True,
                                                )
