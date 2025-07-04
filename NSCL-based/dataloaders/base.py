import torch
import torchvision
from torchvision import transforms
from PIL import Image

from .wrapper import CacheClassLabel
import pickle
import os


def MNIST(dataroot, train_aug=False):
    # Add padding to make 32x32
    normalize = transforms.Normalize(
        mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        # transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR10(dataroot, train_aug=False):
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False):
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def TinyImageNet(dataroot, train_aug=False):
    traindir = os.path.join(dataroot, 'train')
    valdir = os.path.join(dataroot, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=traindir,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.ImageFolder(
        root=valdir,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)
    return train_dataset, val_dataset


class preMiniImageNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(preMiniImageNet, self).__init__()
        self.transform = transform
        if train:
            self.name = 'train'
        else:
            self.name = 'test'
        root = os.path.join(root, 'miniimagenet')
        with open(os.path.join(root, '{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['images']
        self.labels = data_dict['labels']
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.labels[i]
        if not torch.is_tensor(img):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label
    

def MiniImageNet(dataroot, train_aug=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # the transform is following TRGP
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    train_dataset = preMiniImageNet(dataroot, transform=train_transform)
    val_dataset = preMiniImageNet(dataroot, train=False, transform=val_transform)
    train_dataset = CacheClassLabel(train_dataset)
    val_dataset = CacheClassLabel(val_dataset)
    
    return train_dataset, val_dataset