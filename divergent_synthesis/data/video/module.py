import os, sys, pdb
from typing import Type
sys.path.append('..')
import torch, torchvision as tv
from divergent_synthesis.data.video import VideoDataset, VideoTransform, dataset
from divergent_synthesis.utils import checklist
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class Dequantize(torch.nn.Module):
    def __init__(self, range=[0, 255]):
        super().__init__()
        self.range = range

    def __repr__(self):
        return "Dequantize(%s, %s)"%(self.range[0], self.range[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand_like(x, device=x.device) * (1 / self.range[1])

tv.transforms.Dequantize = Dequantize

def parse_transforms(transform_list):
    if transform_list is None:
        return None
    transform_list = checklist(transform_list)
    current_transforms = [tv.transforms.ConvertImageDtype(torch.get_default_dtype())]
    for t in transform_list:
        transform_tmp = getattr(tv.transforms, t['type'])(*t.get('args', tuple()), **t.get('kwargs', {}))
        current_transforms.append(transform_tmp)
    return tv.transforms.Compose(current_transforms)

def parse_augmentations(augmentation_list):
    if augmentation_list is None:
        return []
    augmentation_list = checklist(augmentation_list)
    current_augmentations = []
    for t in augmentation_list:
        augmentation_tmp = getattr(tv.transforms, t['type'])(*t.get('args'), **t.get('kwargs', {}))
        current_augmentations.append(augmentation_tmp)
    return current_augmentations


class VideoDataLoader(DataLoader):
    def __iter__(self, *args, **kwargs):
        for x, y in super().__iter__():
            x = x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
            yield x, y

class VideoDataModule(LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dataset = None
        self.dataset_args = dict(config.dataset)
        self.transform_args = config.get('transforms', {})
        self.augmentation_args = config.get('augmentations', [])
        self.loader_args = config.get('loader', {})
        self.partition_balance = config.get('partition_balance', [0.8, 0.2])
        self.single_file = config.get('single_file', True)
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.import_datasets()

    def load_dataset(self, dataset_args, transform_args, augmentation_args, make_partitions=False):
        transforms = parse_transforms(transform_args.get('transforms'))
        self.transforms = VideoTransform()
        self.full_transforms = transforms
        # import augmentations
        augmentations = parse_augmentations(augmentation_args)
        # create dataset
        dataset = VideoDataset(**dataset_args, transforms=transforms, augmentations=augmentations)
        # set partitions
        if make_partitions:
            dataset.make_partitions(['train', 'valid'], self.partition_balance)
        # set sequence export
        if dataset_args.get('sequence'):
            dataset.drop_sequences(dataset_args['sequence'].get('length'),
                                   dataset_args['sequence'].get('mode', "random"))
        return dataset

    def import_datasets(self, stage = None):
        # transforms
        self.dataset = self.load_dataset(self.dataset_args, self.transform_args, self.augmentation_args, make_partitions=True)
        self.train_dataset = self.dataset.retrieve('train')
        self.valid_dataset = self.dataset.retrieve('valid')

    @property
    def shape(self):
        if self.single_file:
            return tuple(self.dataset[0][0].shape[1:])
        else:
            return tuple(self.dataset[0][0].shape)

    # return the dataloader for each split
    def train_dataloader(self, **kwargs):
        loader_args = {**self.loader_args, **kwargs}
        loader_train = VideoDataLoader(self.train_dataset, **loader_args)
        return loader_train

    def val_dataloader(self, batch_size=None, **kwargs):
        loader_args = {**self.loader_args, **kwargs}
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_val = VideoDataLoader(self.valid_dataset, **loader_args)
        return loader_val

    def test_dataloader(self, batch_size=None, **kwargs):
        if self.test_dataset is None:
            return None
        loader_args = {**self.loader_args, **kwargs}
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_test = VideoDataLoader(self.test_dataset, **loader_args)
        return loader_test

