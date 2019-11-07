from torchvision.datasets import CIFAR10
from torchvision.transforms import Pad, RandomAffine


class ScatteredCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pad=32, translate=0.4):
        super().__init__(root, train=train, target_transform=target_transform, download=download)

        self._transform = transform

        self.p = pad
        self.sz = (2 * self.p) + 32

        self._pad = Pad(pad)
        self._affine = RandomAffine(0, translate=(translate, translate))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img = self._pad(img)
        img = self._affine(img)

        if self._transform is not None:
            img = self._transform(img)

        return img, target
