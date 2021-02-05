import os
import argparse

import numpy as np 
import matplotlib.pyplot as plt 

from PIL import Image

import torch
import torch.nn as nn 

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets

from model import Unet

if __name__ == '__main__':
    # Hyper parameter argparser
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
    parser.add_argument('--batch_size', default=4, type=float, dest='batch_size')
    parser.add_argument('--num_epoch', default=100, type=float, dest='num_epoch')

    parser.add_argument('--data_dir', default="./datasets", type=str, dest='data_dir')
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, dest='ckpt_dir')
    parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
    parser.add_argument('--result_dir', default='./result', type=str, dest='result_dir')

    parser.add_argument('--mode', default='train', type=str, dest='mode')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    u_net = Unet().to(device)
    
    # loss
    loss = nn.BCEWithLogitsLoss().to(device)

    # Optimizer.
    optim = torch.optim.Adam(u_net.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(10):
        u_net.train()
        