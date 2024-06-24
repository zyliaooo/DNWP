"""
Comparison algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ResUnet (Unet with RB)
class ResUnet(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()

        if dims is None:
            dims = [16, 32, 64, 128]
        self.scale = len(dims)
        self.before = []

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=3, padding=1, padding_mode='zeros', bias=False),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv3d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        self.down2 = nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        self.down3 = nn.Conv3d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, padding_mode='zeros')

        self.en1 = RB1(dims[0])
        self.en2 = RB1(dims[1])
        self.en3 = RB1(dims[2])
        self.en4 = RB1(dims[3])

        self.up3 = nn.ConvTranspose3d(dims[3], dims[2], 3, 2, 1, output_padding=1, bias=False)
        self.up2 = nn.ConvTranspose3d(dims[2], dims[1], 3, 2, 1, output_padding=1, bias=False)
        self.up1 = nn.ConvTranspose3d(dims[1], dims[0], 3, 2, 1, output_padding=1, bias=False)

        # Decoder
        self.de3 = RB1(dims[2], dims[2] * 2, True)
        self.de2 = RB1(dims[1], dims[1] * 2, True)
        self.de1 = RB1(dims[0], dims[0] * 2, True)

        self.head = nn.Conv3d(dims[0], 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.en1(x)
        x = self.down1(x1)
        x2 = self.en2(x)
        x = self.down2(x2)
        x3 = self.en3(x)
        x = self.down3(x3)
        x = self.en4(x)

        x = self.up3(x)
        x = torch.cat((x3, x), dim=1)
        del x3
        x = self.de3(x)

        x = self.up2(x)
        x = torch.cat((x2, x), dim=1)
        del x2
        x = self.de2(x)

        x = self.up1(x)
        x = torch.cat((x1, x), dim=1)
        del x1
        x = self.de1(x)

        x = self.head(x)

        return x


class RB1(nn.Module):
    def __init__(self, dim, in_dim=None, is_shortcut=False):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.is_shortcut = is_shortcut
        self.conv1 = nn.Conv3d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky2 = nn.ReLU(inplace=True)
        if is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, dim, 1),
            )

    def forward(self, x):
        identity = x
        x = self.leaky1(self.conv1(x))
        x = self.leaky2(self.conv2(x))
        if self.is_shortcut:
            identity = self.shortcut(identity)
        x = identity + x
        return x


# DLPU (Unofficial implementation)
class dlpu(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()

        if dims is None:
            dims = [16, 32, 64, 128]
        self.scale = len(dims)
        self.before = []

        # Encoder
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=3, padding=1, padding_mode='zeros', bias=False),
            nn.ReLU(inplace=True),
        )
        self.downsample_layers.append(stem)
        for i in range(self.scale - 1):
            downsample_layer = nn.Sequential(
                nn.MaxPool3d(2, 2),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                nn.ReLU(inplace=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(self.scale):
            stage = nn.Sequential(
                RB2(dims[i], dims[i]),
                nn.Conv3d(dims[i], dims[i], kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)

        # Decoder
        self.ups = nn.ModuleList()
        for j in range(self.scale - 1):
            up1 = nn.ConvTranspose3d(dims[self.scale - 1 - j], dims[self.scale - 2 - j], 3, 2, 1, output_padding=1,
                                     bias=False)
            self.ups.append(up1)

        self.decoders = nn.ModuleList()
        for j in range(self.scale - 1):
            decoder = nn.Sequential(
                nn.Conv3d(dims[self.scale - 2 - j] * 2, dims[self.scale - 2 - j], kernel_size=3, padding=1,
                          padding_mode='zeros', bias=False),
                nn.ReLU(inplace=True),
                RB2(dims[self.scale - 2 - j], dims[self.scale - 2 - j]),
                nn.Conv3d(dims[self.scale - 2 - j], dims[self.scale - 2 - j], kernel_size=3, padding=1,
                          padding_mode='zeros', bias=False),
                nn.ReLU(inplace=True),
            )
            self.decoders.append(decoder)

        self.head = nn.Sequential(
            nn.Conv3d(dims[0], 1, kernel_size=1, bias=False),
        )

    def forward_features(self, x):
        for i in range(self.scale):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            self.before.append(x)
        self.before.pop()

        for j in range(self.scale - 1):
            x = self.ups[j](x)
            before = self.before.pop()
            x = torch.cat((before, x), dim=1)
            x = self.decoders[j](x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class RB2(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.leaky1(self.conv1(x))
        x = self.leaky2(self.conv2(x))
        x = identity + x
        return x


# PU-M-Net (Unofficial implementation)
class pumnet(nn.Module):
    def __init__(self, dims: list = None):
        super().__init__()

        if dims is None:
            dims = [16, 32, 64, 128]
        self.scale = len(dims)
        self.before = []
        self.outputs = []
        self.mpouts = []

        self.MPs = nn.ModuleList()
        for _ in range(self.scale - 1):
            self.MPs.append(nn.MaxPool3d(2, 2))
        # Encoder
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=3, padding=1, padding_mode='zeros', bias=False),
            nn.ReLU(inplace=True),
        )
        self.downsample_layers.append(stem)
        for i in range(self.scale - 1):
            downsample_layer = nn.Sequential(
                nn.MaxPool3d(2, 2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(self.scale):
            if i == 0:
                stage = nn.Sequential(
                    RB3(dims[0], dims[0]),
                    nn.Conv3d(dims[0], dims[0], kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                    nn.ReLU(inplace=True),
                )
            else:
                stage = Pblock(dims[i-1]+1, dims[i], dims[i-1]+1+dims[i])
            self.stages.append(stage)

        # Decoder
        self.ups = nn.ModuleList()
        for j in range(self.scale - 1):
            up1 = nn.ConvTranspose3d(dims[self.scale - 1 - j], dims[self.scale - 2 - j], 3, 2, 1, output_padding=1,
                                     bias=False)
            self.ups.append(up1)

        self.decoders = nn.ModuleList()
        for j in range(self.scale - 1):
            decoder = Pblock(dims[self.scale - 2 - j] * 2, dims[self.scale - 2 - j], dims[self.scale - 2 - j] * 3)
            self.decoders.append(decoder)

        # if line 261 is used, sum(dims[:])
        self.head = nn.Sequential(
            nn.Conv3d(sum(dims[:-1]), 1, kernel_size=1, bias=False),
        )

    def forward_features(self, x):
        for i in range(self.scale):
            x = self.downsample_layers[i](x)
            if i != 0:
                mpout = self.mpouts.pop(0)
                x = torch.cat([mpout, x], dim=1)
            x = self.stages[i](x)
            self.before.append(x)
        self.before.pop()
        # self.outputs.append(x)    # more memory usage, if used.

        for j in range(self.scale - 1):
            x = self.ups[j](x)
            before = self.before.pop()
            x = torch.cat((before, x), dim=1)
            x = self.decoders[j](x)
            self.outputs.append(x)

    def forward(self, x):
        mpout = x
        for i in range(self.scale - 1):
            mpout = self.MPs[i](mpout)
            self.mpouts.append(mpout)
        self.forward_features(x)
        for d in range(len(self.outputs)):
            self.outputs[d] = _upsample_like(self.outputs[d], x)
        del x
        del mpout
        out = torch.cat(self.outputs[:], dim=1)
        self.outputs.clear()

        out = self.head(out)

        return out


class RB3(nn.Module):
    def __init__(self, dim, in_dim=None, is_shortcut=False):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.is_shortcut = is_shortcut
        self.conv1 = nn.Conv3d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky2 = nn.ReLU(inplace=True)
        if is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, dim, 1),
            )

    def forward(self, x):
        identity = x
        x = self.leaky1(self.conv1(x))
        x = self.leaky2(self.conv2(x))
        if self.is_shortcut:
            identity = self.shortcut(identity)
        x = identity + x
        return x


def _upsample_like(src, tar):
    x = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
    return x


class Pblock(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv3d(dim1, dim2, kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                    nn.ReLU(inplace=True),
                )
        self.conv2 = RB3(dim2, dim2)
        self.conv3 = nn.Sequential(
                    nn.Conv3d(dim3, dim2, kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                    nn.ReLU(inplace=True),
                )

    def forward(self, x):
        x_short = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([x_short, x], dim=1)
        x = self.conv3(x)
        return x
