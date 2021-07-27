import os
import sys
import pathlib
import time
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torch.distributions import Categorical
from apex import amp
from net import build_model
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
from utils.plotter import plot_results
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nclasses', type=int, required=True)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()
    init_seeds()
    device = set_device(args.gpu)

    transform = build_transform(rescale_size=448, crop_size=448)
    dataset = build_webfg_dataset(os.path.join('Datasets', args.dataset), transform['train'], transform['test'])
    net = build_model(arch='clar-resnet50', n_classes=args.nclasses, pretrained=True, reduction_factor=2)
    net = net.to(device=device)
    net.load_state_dict(torch.load(args.model_path))
    test_loader = DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    test_accuracy = evaluate(test_loader, net, device)
    print(f'Test accuracy: {test_accuracy:.3f}')
