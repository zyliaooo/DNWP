import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from Pro_utility import RB, RBBN
from Pro_dcn import DB


train_data_path = r"./XX/XX"
val_data_path = r"./XX/XX"

learning_rate = 0.0001
batch_size = 4
epochs = 15
version = 'XX'
lr_mult = 0.1


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
                # select for val
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
            # mag = mag / np.max(mag)  # optional
            mask = pre_image['mask']
            return mag, mask
        elif self.task_tag == 'classify':
            phase = pre_image['phase']
            clc = pre_image['clc']
            return phase, clc  # 1 means need to be masked


# model
class DGNetM(nn.Module):
    """
    Args:
        dims (int): Feature dimension at each stage. Default: [16, 32, 64]
    """

    def __init__(self, dims: list = None, segnum=1):
        super().__init__()

        if dims is None:
            dims = [16, 32, 64]
        self.before = []

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(dims[0]),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.down1 = nn.Conv3d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.down2 = nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, padding_mode='replicate')

        self.en1 = RBBN(dims[0])
        self.en2 = RBBN(dims[1])
        self.en3 = DB(dims[2])

        # Decoder
        self.de2 = RBBN(dims[1], dims[2] + dims[1])
        self.de1 = RBBN(dims[0], dims[1] + dims[0])
        self.head = nn.Conv3d(dims[0], segnum, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.en1(x)
        self.before.append(x)
        x = self.down1(x)
        x = self.en2(x)
        self.before.append(x)
        x = self.down2(x)
        x = self.en3(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((self.before[1], x), dim=1)
        del self.before[1]
        x = self.de2(x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((self.before[0], x), dim=1)
        del self.before[0]
        x = self.de1(x)
        x = self.head(x)
        return x


def train_loop(dataloader):
    for batch, (x, y1) in enumerate(dataloader):
        x = x.type(torch.FloatTensor).unsqueeze(1).cuda()
        y1 = y1.type(torch.FloatTensor).unsqueeze(1).cuda()

        pred = model(x)
        loss = loss_fn(pred, y1)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            train_loss = loss.item()
            print(f"train: {train_loss:>7f} ")


def val_loop(dataloader, epoch, version):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for batch, (x, y1) in enumerate(dataloader):
            x = x.type(torch.FloatTensor).unsqueeze(1).cuda()
            y1 = y1.type(torch.FloatTensor).unsqueeze(1).cuda()
            pred = model(x)
            val_loss += loss_fn(pred, y1).item()
        val_loss /= num_batches

        # save
        torch.save(model, f'./XX/{version}_{epoch}.pkl')
        print(f"val: {val_loss:>7f}")
        print(f"Model Saved Successfully!")


if __name__ == "__main__":
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataload
    training_data = MyDataset(train_data_path, 'mask')
    valing_data = MyDataset(val_data_path, 'mask', train_tag=False)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(valing_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)

    # model create
    model = DGNetM()
    model = nn.DataParallel(model)
    model.cuda()

    # Trainable part
    print("\n Trainable part of the model is：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("\n -------------------------------")

    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')

    # optimizer
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
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[11, 21], gamma=0.5, last_epoch=-1)

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
