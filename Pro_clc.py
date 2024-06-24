import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, einsum
import scipy.io as scio
import math
import numpy as np
import random
import time
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


train_data_path = r"./XX/XX"
val_data_path = r"./XX/XX"

learning_rate = 0.0001
batch_size = 64
epochs = 10
version = 'XX'


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
        print(f" \n We use {len(self.img_path)} train samples now. \n")

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        image_single_path = self.img_path[index]
        pre_image = scio.loadmat(image_single_path)
        assert self.task_tag in ['phase', 'mask', 'classify'], "task_tag should be 'phase' 'classify' or 'mask'"
        if self.task_tag == 'phase':
            data = pre_image['data']
            label = pre_image['label']
            return data, label
        elif self.task_tag == 'mask':
            mag = pre_image['mag']
            mag = mag / np.max(mag)
            mask = pre_image['mask']
            return mag, mask
        elif self.task_tag == 'classify':
            phase = pre_image['phase']
            clc = pre_image['clc']
            phase_min = phase.min()
            phase = np.where(phase > phase_min, phase, 0)
            return phase, clc  # 1 means should be masked


# model
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 768, img_size: int = 192):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # Conv replace Linear projection
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # (img_size // patch_size)**2 + 1(cls token)
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 4, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, n_classes: int = 2):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

# patch_size: int = 16 emb_size: int = 32
class DGNetC(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 32,
                 img_size: int = 192,
                 depth: int = 1,
                 n_classes: int = 1,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# patch_size: int = 16 emb_size: int = 64
class DGNetC2(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 64,
                 img_size: int = 192,
                 depth: int = 1,
                 n_classes: int = 1,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# patch_size: int = 16 emb_size: int = 128
class DGNetC3(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 128,
                 img_size: int = 192,
                 depth: int = 1,
                 n_classes: int = 1,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# patch_size: int = 8 emb_size: int = 64
class DGNetC4(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 8,
                 emb_size: int = 64,
                 img_size: int = 192,
                 depth: int = 1,
                 n_classes: int = 1,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# patch_size: int = 32 emb_size: int = 64
class DGNetC5(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 32,
                 emb_size: int = 64,
                 img_size: int = 192,
                 depth: int = 1,
                 n_classes: int = 1,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


def train_loop(dataloader):
    for batch, (x, y1) in enumerate(dataloader):
        x = x.type(torch.FloatTensor).unsqueeze(1).cuda()
        y1 = y1.type(torch.FloatTensor).squeeze(-1).cuda()

        pred = model(x)
        loss = loss_fn(pred, y1)

        # backwrad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            train_loss = loss.item()
            print(f"train: {train_loss:>7f} ")


def val_loop(dataloader, epoch, version):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for batch, (x, y1) in enumerate(dataloader):
            x = x.type(torch.FloatTensor).unsqueeze(1).cuda()
            y1 = y1.type(torch.FloatTensor).squeeze(-1).cuda()
            pred = model(x)
            val_loss += loss_fn(pred, y1).item()
        val_loss /= num_batches

        torch.save(model, f'./XX/{version}_{epoch}.pkl')
        print("Model Saved Successfully!")
        print(f"val: {val_loss:>7f}")


if __name__ == "__main__":
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataload
    training_data = MyDataset(train_data_path, 'classify')
    valing_data = MyDataset(val_data_path, 'classify', train_tag=False)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(valing_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # model create
    model = DGNetC2()
    model = model.cuda()

    # Trainable part
    print("\n Trainable part of the model is：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("\n -------------------------------")

    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')

    # optimizers
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0,
                                  amsgrad=False)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=10)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5, last_epoch=-1)

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
