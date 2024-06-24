"""
BFANet and ProNet
"""
import os
import skimage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.io as scio
import math
import numpy as np
import random
import time
from Pro_utility import RB, RBBN, RLoss, _upsample_like, OSA, G_operator2
from Pro_dcn import DB
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

version = 'XX'
lr_mult = 0.1

train_data_path = r"./XX/XX"
val_data_path = r"./XX/XX"

learning_rate = 0.0001
batch_size = 4
epochs = 25
gate_value = 0


# Dataset
class MyDataset(Dataset):
    def __init__(self, img_dir, task_tag, train_tag=True):
        self.task_tag = task_tag
        self.img_dir = img_dir
        self.img_path = [os.path.join(self.img_dir, filename) for filename in os.listdir(self.img_dir)]
        sample = []
        for filename in os.listdir(self.img_dir):
            # select for train
            if train_tag:
                tag = ('XX' not in filename)
            else:
                tag = ('XX' in filename)
            if tag:
                name = os.path.join(self.img_dir, filename)
                sample.append(name)
        self.img_path = sample
        print(f" \n We use {len(self.img_path)} samples now. \n")

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        image_single_path = self.img_path[index]
        # print(image_single_path)
        pre_image = scio.loadmat(image_single_path)
        assert self.task_tag in ['phase', 'mask', 'classify'], "task_tag should be 'phase' 'classify' or 'mask'"
        if self.task_tag == 'phase':
            data = pre_image['data']
            label = pre_image['label']
            return data, label
        elif self.task_tag == 'mask':
            mag = pre_image['mag']
            mag = mag / np.max(mag)     # optional
            mask = pre_image['mask']
            return mag, mask
        elif self.task_tag == 'classify':
            phase = pre_image['phase']
            clc = pre_image['clc']
            return phase, clc  # 1 means should to be masked


# models
class P1(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()

        if dims is None:
            dims = [8, 16, 32, 64]
        self.scale = len(dims)
        self.before = []

        # Encoder
        self.stem = torch.load('./XX/XX').module.stem
        self.GOP = torch.load('./XX/XX').module.GOP

        self.down1 = nn.Conv3d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.down2 = nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.down3 = nn.Conv3d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, padding_mode='replicate')

        self.en1 = RB(dims[0], 16)
        self.en2 = RB(dims[1])
        self.en3 = RB(dims[2])
        self.en4 = RB(dims[3])

        # Decoder
        self.de3 = RB(dims[2], dims[2] + dims[3])
        self.de2 = RB(dims[1], dims[1] + dims[2])
        self.de1 = RB(dims[0], dims[0] + dims[1])

        self.head = nn.Conv3d(dims[0], 1, kernel_size=1, bias=False)

    def forward(self, x):

        raw = x
        x = self.stem(x)
        gop = self.GOP(raw, x)
        gop.detach_()
        x1 = self.en1(gop)
        x = self.down1(x1)
        x2 = self.en2(x)
        x = self.down2(x2)
        x3 = self.en3(x)
        x = self.down3(x3)
        x = self.en4(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x3, x), dim=1)
        del x3
        x = self.de3(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x2, x), dim=1)
        del x2
        x = self.de2(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x1, x), dim=1)
        del x1
        x = self.de1(x)

        x = self.head(x)

        return x + raw


class P12(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()

        if dims is None:
            dims = [8, 16, 32, 64]
        self.scale = len(dims)
        self.before = []

        # Encoder
        self.stem = torch.load('./XX/XX').module.stem
        self.GOP = torch.load('./XX/XX').module.GOP

        self.down1 = nn.Conv3d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.down2 = nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.down3 = nn.Conv3d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, padding_mode='replicate')

        self.en1 = RB(dims[0], 16)
        self.en2 = RB(dims[1])
        self.en3 = RB(dims[2])
        self.en4 = RB(dims[3])

        # Decoder
        self.de3 = RB(dims[2], dims[2] + dims[3])
        self.de2 = RB(dims[1], dims[1] + dims[2])
        self.de1 = RB(dims[0], dims[0] + dims[1])

        self.head = nn.Conv3d(dims[0], 1, kernel_size=1, bias=False)

    def forward(self, x):

        raw = x
        x = self.stem(x)
        gop = self.GOP(raw, x)
        # gop.detach_()
        x1 = self.en1(gop)
        x = self.down1(x1)
        x2 = self.en2(x)
        x = self.down2(x2)
        x3 = self.en3(x)
        x = self.down3(x3)
        x = self.en4(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x3, x), dim=1)
        del x3
        x = self.de3(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x2, x), dim=1)
        del x2
        x = self.de2(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x1, x), dim=1)
        del x1
        x = self.de1(x)

        x = self.head(x)

        return x + raw


class R(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()

        if dims is None:
            dims = [16, 32, 64, 128]
        self.scale = len(dims)

        # Encoder
        self.downsample_layers = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv3d(1, dims[0] - 1, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.GOP = G_operator2()
        self.downsample_layers.append(self.GOP)
        self.MP1 = nn.MaxPool3d(2, 2)
        self.MP2 = nn.MaxPool3d(2, 2)
        for i in range(self.scale - 1):
            downsample_layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        stage = OSA(dims[0], dims[0])
        self.stages.append(stage)
        stage = OSA(dims[1], dims[1] + 1)
        self.stages.append(stage)
        stage = OSA(dims[2], dims[2] + 1)
        self.stages.append(stage)

        # Dynamic
        self.Dconv = DB(dims[-1], dims[-1])

        # Decoder
        self.decoders = nn.ModuleList()
        decoder = RB(dims[2], dims[3] + dims[2])
        self.decoders.append(decoder)
        decoder = RB(dims[1], dims[2] + dims[1])
        self.decoders.append(decoder)
        decoder = RB(dims[0], dims[1] + dims[0])
        self.decoders.append(decoder)

        # multi-output
        self.head2 = nn.Conv3d(dims[1], dims[0], kernel_size=1)
        self.head3 = nn.Conv3d(dims[2], dims[0], kernel_size=1)
        self.OutRB = nn.Sequential(
            RB(dims[0], dims[0] * 3),
            nn.Conv3d(dims[0], 1, kernel_size=1),
        )

    def forward_features(self, raw, x):
        mp1 = self.MP1(raw)
        mp2 = self.MP2(mp1)

        x = self.downsample_layers[0](raw, x)
        be1 = self.stages[0](x)
        x = self.downsample_layers[1](be1)
        x = torch.cat([mp1, x], dim=1)
        be2 = self.stages[1](x)
        x = self.downsample_layers[2](be2)
        x = torch.cat([mp2, x], dim=1)
        be3 = self.stages[2](x)
        x = self.downsample_layers[3](be3)

        x = self.Dconv(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((be3, x), dim=1)
        del be3
        d3 = self.decoders[0](x)

        x = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((be2, x), dim=1)
        del be2
        d2 = self.decoders[1](x)

        x = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((be1, x), dim=1)
        del be1
        d1 = self.decoders[2](x)

        return d1, d2, d3

    def forward(self, x):

        raw = x
        x = self.stem(x)
        d1, d2, d3 = self.forward_features(raw, x)
        # side output
        d2 = self.head2(d2)
        d3 = self.head3(d3)
        d2 = _upsample_like(d2, x)
        d3 = _upsample_like(d3, x)
        x = torch.cat([d1, d2, d3], dim=1)
        del d1, d2, d3
        x = self.OutRB(x)

        return x


def train_loop(dataloader):
    for batch, (x, y2) in enumerate(dataloader):
        x = x.type(torch.FloatTensor).unsqueeze(1).cuda()
        y2 = y2.type(torch.FloatTensor).unsqueeze(1).cuda()
        pred = model(x)

        # for P
        # complexity = (y2 - x) / (2 * math.pi)
        # complexity = complexity.max() - complexity.min() + 1
        # if complexity < 6:   # K_delta set to 6
        #     factor = 1.5
        # else:
        #     factor = 1

        # for R
        factor = 1

        train_loss1 = factor * loss_fn(pred, y2)         # factor for P
        train_loss2 = factor * RLoss(pred, y2)           # factor for P
        loss = train_loss1 + 0.3 * train_loss2

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            print(f"train1: {train_loss1.item():>7f} train2: {0.3 * train_loss2.item():>7f}")


def val_loop(dataloader, epoch, version):

    num_batches = len(dataloader)
    val_loss1 = 0
    val_loss2 = 0

    with torch.no_grad():
        for batch, (x, y2) in enumerate(dataloader):
            x = x.type(torch.FloatTensor).unsqueeze(1).cuda()
            y2 = y2.type(torch.FloatTensor).unsqueeze(1).cuda()
            pred = model(x)

            # for P
            # complexity = (y2 - x) / (2 * math.pi)
            # complexity = complexity.max() - complexity.min() + 1
            # if complexity < 6:
            #     factor = 1.5
            # else:
            #     factor = 1

            # for R
            factor = 1

            val_loss1 += factor * loss_fn(pred, y2).item() / num_batches    # factor for P
            val_loss2 += factor * RLoss(pred, y2).item() / num_batches      # factor for P
        print(f"val1: {val_loss1:>7f} val2: {0.3 * val_loss2:>7f}")
        val_loss = val_loss1 + 0.3 * val_loss2

        # Save
        torch.save(model, f'./XX/{version}_{epoch}.pkl')
        print(f"Model {version} Saved Successfully!")


if __name__ == "__main__":

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataload
    training_data = MyDataset(train_data_path, 'phase')
    valing_data = MyDataset(val_data_path, 'phase', train_tag=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                                  pin_memory=True)
    val_dataloader = DataLoader(valing_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True,
                                pin_memory=True)

    # model create
    model = R()
    model = nn.DataParallel(model)
    model.cuda()

    # trainable part
    print("\n Trainable part of the model is：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("\n -------------------------------")

    # loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # optimizers
    params = list(model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if 'offset' not in n], 'lr': learning_rate},
        {'params': [p for n, p in params if 'offset' in n], 'lr': learning_rate * lr_mult},
    ]
    for n, p in params:
        if 'offset' in n:
            print(n)

    optimizer = torch.optim.AdamW(param_group,
                                  lr=learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0,
                                  amsgrad=False)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=epochs)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[11, 16, 21], gamma=0.5, last_epoch=-1)

    # loop
    for t in range(epochs):
        if t == 0:
            best_loss = val_loop(val_dataloader, t + 1, version)
        # train
        print(f"Epoch {t + 1}\n-------------------------------")
        model.train()
        train_loop(train_dataloader)
        scheduler.step()
        print(f"now learning rate is: {optimizer.state_dict()['param_groups'][0]['lr']:>7f}\n")
        # eval
        model.eval()
        val_loop(val_dataloader, t + 1, version)

    print("Done!")
