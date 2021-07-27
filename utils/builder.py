import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.noisy_cifar import NoisyCIFAR100
from data.image_folder import IndexedImageFolder
import numpy as np
import cv2


# dataset --------------------------------------------------------------------------------------------------------------------------------------------
def build_transform(rescale_size=512, crop_size=448):
    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return {'train': train_transform, 'test': test_transform,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform}


def build_cifar100n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}


def build_webfg_dataset(root, train_transform, test_transform):
    train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test_data = IndexedImageFolder(os.path.join(root, 'val'), transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}


# optimizer, scheduler -------------------------------------------------------------------------------------------------------------------------------
def build_optimizer(params, lr, weight_decay, opt='sgd'):
    if opt == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif opt == 'sgd':
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    else:
        raise AssertionError(f'{opt} optimizer is not supported!')


def build_sgd_optimizer(params, lr, weight_decay, nesterov=True):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)


def build_adam_optimizer(params, lr):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999))


def build_cosine_lr_scheduler(optimizer, total_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

