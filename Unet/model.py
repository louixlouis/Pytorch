import os

import torch
import torch.nn as nn
import torch.nn.functional as F 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()           # subclass에서 base-class의 내용을 overwriting해서 사용하고 싶을 때.
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        #self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)
        
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.enc_1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.enc_1_2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.enc_2_1 = ConvBlock(in_channels=64,  out_channels=128)
        self.enc_2_2 = ConvBlock(in_channels=128,  out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc_3_1 = ConvBlock(in_channels=128, out_channels=256)
        self.enc_3_2 = ConvBlock(in_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc_4_1 = ConvBlock(in_channels=256, out_channels=512)
        self.enc_4_2 = ConvBlock(in_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)

        # bottle_neck
        self.enc_5 = ConvBlock(in_channels=512, out_channels=1024)
        self.dec_5 = ConvBlock(in_channels=1024, out_channels=512)

        #self.up_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_4_1 = ConvBlock(in_channels=1024, out_channels=512)    # 512 + 512
        self.dec_4_2 = ConvBlock(in_channels=512, out_channels=256)
        
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_3_1 = ConvBlock(in_channels=512, out_channels=256)     # 256 + 256
        self.dec_3_2 = ConvBlock(in_channels=256, out_channels=128)
        
        self.up_2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_2_1 = ConvBlock(in_channels=256, out_channels=128)     # 128 + 128
        self.dec_2_2 = ConvBlock(in_channels=128, out_channels=64)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_1_1 = ConvBlock(in_channels=128, out_channels=64)      # 64 + 64
        self.dec_1_2 = ConvBlock(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def save_checkpoint(self, checkpoint_dir, net, optim, epoch, iteration):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_filename = 'epoch_%s_iteration_%s.pth' % (epoch, iteration)
        save_path = os.path.join(checkpoint_dir, model_filename)
        torch.save({
            'epoch':epoch,
            'model_state_dict':net.state_dict(),
            'optim_state_dict':optim.state_dict()},
            save_path)

    def forwad(self, x):
        # Encoder
        enc_1 = self.enc_1_2(self.enc_1_1(x))
        pool_1 = self.pool_1(enc_1)
        enc_2 = self.enc_2_2(self.enc_2_1(pool_1))
        pool_2 = self.pool_2(enc_2)
        enc_3 = self.enc_3_2(self.enc_3_1(pool_2))
        pool_3 = self.pool_3(enc_3)
        enc_4 = self.enc_4_2(self.enc_4_1(pool_3))
        pool_4 = self.pool_4(enc_4)
        # Bottle_neck
        enc_5 = self.enc_5(pool_4)
        dec_5 = self.dec_5(enc_5)
        # Decoder                                                                                                                                                                                                                                                                                                                                                    ssssss
        up_4 = self.up_4(dec_5)
        skip_4 = torch.cat([up_4, pool_4], dim=1)
        dec_4 = self.dec_4_2(self.dec_4_1(skip_4))
        
        up_3 = self.up_3(dec_4)
        skip_3 = torch.cat([up_3, pool_3], dim=1)
        dec_3 = self.dec_3_2(self.dec_3_1(skip_3))
        up_2 = self.up_2(dec_3)
        skip_2 = torch.cat([up_2, pool_2], dim =1)
        dec_2 = self.dec_2_2(self.dec_2_1(skip_2))
        up_1 = self.up_1(dec_2)
        skip_1 = torch.cat([up_1, pool_1], dim=1)
        dec_1 = self.dec_1_2(self.dec_1_1(skip_1))
        x = self.fc(dec_1)
        
        return x

