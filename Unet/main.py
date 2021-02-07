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
from dataset import CasiaWebFace

if __name__ == '__main__':
    # Hyper parameter argparser
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-3, type=float, dest='lr') 
    parser.add_argument('--batch_size', default=20, type=float, dest='batch_size')
    parser.add_argument('--num_epoch', default=10, type=float, dest='num_epoch')

    parser.add_argument('--data_dir', default="./datasets/casia", type=str, dest='data_dir')
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, dest='ckpt_dir')
    parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
    parser.add_argument('--result_dir', default='./result', type=str, dest='result_dir')

    parser.add_argument('--mode', default='train', type=str, dest='mode')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Dataloader
    transform = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize
            transforms.RandomHorizontalFlip(p=0.5),                 # Horizontal flip
            transforms.ToTensor(),                                  # To pytorch tensor
        ]
    )
            
    dataset_train = CasiaWebFace(
        mask_root = os.path.join(args.data_dir, 'mask_train'), 
        label_root = os.path.join(args.data_dir, 'label_train'), 
        transform = transform)    

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size)

    num_data = len(dataset_train)
    num_epoch = np.ceil(num_data / args.batch_size)

    print(f'Number of data:     {num_data}')
    print(f'Number of epochs:   {num_epoch}')
    

    # Model load.
    u_net = Unet().to(device)
    
    # loss
    loss = nn.BCEWithLogitsLoss().to(device)

    # Optimizer.
    optim = torch.optim.Adam(u_net.parameters(), lr=args.lr)

    # 기타 function 설정
    to_numpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환
    denorm = lambda x, mean, std : (x * std) + mean
    classifier = lambda x :  1.0 * (x > 0.5)  # threshold 0.5 기준으로 indicator function으로 classifier 구현

    # Tensorboard
    tensorboard_train = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))

    # Training loop
    init_epoch = 0
    max_num_iters = num_data // args.batch_size

    for epoch in range(init_epoch, args.num_epoch):
        #u_net.train()
        loss_history = []
        cur_epoch_data_loader_casia = iter(loader_train)
        for i in range(max_num_iters):
            # Forward.
            mask_tensor, label_tensor = next(cur_epoch_data_loader_casia)
            mask_tensor, label_tensor = mask_tensor.to(device). label_tensor.to(device)

            output_tensor = u_net(mask_tensor)

            # Backward.
            optim.zero_grad()
            loss = loss(output_tensor, label_tensor)
            loss.backward()
            optim.step()

            # Save the loss.
            loss_history.append(loss.item())

            # write the result to tensorboad.
            label = to_numpy(label_tensor)
            inputs = to_numpy(denorm(mask_tensor, 0.5, 0.5))
            output = to_numpy(classifier(output_tensor))

            tensorboard_train.add_image('label', label, args.num_epoch * epoch + i, dataformats='NHWC')
            tensorboard_train.add_image('input', inputs, args.num_epoch * epoch + i, dataformats='NHWC')
            tensorboard_train.add_image('output', output, args.num_epoch * epoch + i, dataformats='NHWC')

            print(f'EPOCH : [{epoch+1}]/[{num_epoch}]')
            print(f'ITERATION : [{i+1}]/[{max_num_iters}]')
            print(f'LOSS : {loss}')

            # Save the model.
            if i % 1000 == 0:
                u_net.save_checkpoint(args.checkpoint_dir, u_net, optim, epoch, i)

        tensorboard_train.add_scalar('loss', np.mean(loss_history), epoch)
