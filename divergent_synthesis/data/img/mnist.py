import os, sys, pdb
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
from pytorch_lightning import LightningDataModule
from .transforms import Compose, Rescale, Binarize


class MNIST(datasets.MNIST):
    def __getitem__(self, item):
        x, y = super(MNIST, self).__getitem__(item)
        return x, {'class': y}


class MNISTDataModule(LightningDataModule):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self.data_args = config.get('dataset', {})
        self.polarity = self.data_args.get('polarity', 'unipolar')
        self.loader_args = dict(config.get('loader', {'batch_size': 64}))

    @property
    def shape(self):
        if self.data_args.get('resize'):
            return (1, *self.data_args.resize)
        else:
            return (1, 28, 28)

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage = None):
        # transforms
        transform = []
        if self.data_args.get('resize'):
            transform.append(transforms.Resize(tuple(self.data_args.resize)))
        transform.append(transforms.ToTensor())
        if self.data_args.binary:
            transform.append(Binarize())
        transform.append(Rescale(mode = self.polarity))
        transform = Compose(transform)
        # split dataset
        mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)
        self.transforms = transform

    def train_dataloader(self, **kwargs):
        loader_args = {**self.loader_args, **kwargs}
        loader_train = DataLoader(self.train_dataset(), **loader_args)
        return loader_train

    def val_dataloader(self, **kwargs):
        loader_args = {**self.loader_args, **kwargs}
        loader_args['shuffle'] = False
        loader_val = DataLoader(self.validation_dataset(), **loader_args)
        return loader_val

    def test_dataloader(self, **kwargs):
        if self.test_dataset is None:
            return None
        loader_args = {**self.loader_args, **kwargs}
        loader_args['shuffle'] = False
        loader_test = DataLoader(self.test_dataset(), **loader_args)
        return loader_test

    # utils callback
    def train_dataset(self, **kwargs):
        return self.mnist_train
    def validation_dataset(self, **kwargs):
        return self.mnist_val
    def test_dataset(self, **kwargs):
        return self.mnist_test


class ClassifMNISTDataModule(MNISTDataModule):
    def setup(self, stage=None):
                # transforms
        transform = [
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomAffine(30, translate=(0.2, 0.2), scale=(0.3, 0.3), shear=None, resample=0, fillcolor=0),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.RandomErasing()
            ]),
        ]
        if self.data_args.get('resize'):
            transform.insert(1, transforms.Resize(tuple(self.data_args.resize)))
        if self.data_args.binary:
            transform.append(Binarize())
        transform.append(Rescale(mode = self.polarity))
        transform = Compose(transform)
        # split dataset
        mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)
        self.transforms = transform
