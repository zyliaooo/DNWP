"""
Some components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RB(nn.Module):
    def __init__(self, dim, in_dim=None):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.conv1 = nn.Conv3d(in_dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.leaky1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.leaky2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.leaky3 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.leaky1(self.conv1(x))
        identity = x
        x = self.leaky2(self.conv2(x))
        x = self.conv3(x)
        x = identity + x
        x = self.leaky3(x)
        return x


class RBBN(nn.Module):
    def __init__(self, dim, in_dim=None):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.conv1 = nn.Conv3d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(dim)
        self.leaky1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(dim)
        self.leaky2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(dim)
        self.leaky3 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(self.leaky1(self.conv1(x)))
        identity = x
        x = self.bn2(self.leaky2(self.conv2(x)))
        x = self.conv3(x)
        x = identity + x
        x = self.bn3(self.leaky3(x))
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y


class OSA(nn.Module):
    def __init__(self, dim, indim, is_shortcut=False, padding_mode='replicate'):
        super().__init__()

        self.is_shortcut = is_shortcut
        self.conv0 = nn.Sequential(
            nn.Conv3d(indim, dim, 3, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv111 = nn.Sequential(
            nn.Conv3d(dim * 3, dim, 1),
        )
        if is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(indim, dim, 1),
            )

    def forward(self, x):
        x = self.conv0(x)
        x_identity = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv111(x)
        if self.is_shortcut:
            x_identity = self.shortcut(x_identity)
        x = x + x_identity

        return x


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(channel, channel, kernel_size=1)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


def _upsample_like(src, tar):
    # src = F.upsample(src, size=tar.shape[2:], mode='trilinear')
    x = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
    return x


def to_one_hot(mask, n_class):
    b, _, T, H, W = mask.shape
    y_one_hot = torch.zeros((b, n_class, T, H, W)).cuda()
    y_one_hot = y_one_hot.scatter(1, mask, 1)
    return y_one_hot


def RLoss_L1(generated, real_image):  # b x c x h x w
    b, c, h, w, d = generated.shape
    all = b * c * h * w * d

    true_x_shifted_right = real_image[:, :, 1:, :, :]
    true_x_shifted_left = real_image[:, :, :-1, :, :]
    true_x_gradient = true_x_shifted_right - true_x_shifted_left

    generated_x_shift_right = generated[:, :, 1:, :, :]
    generated_x_shift_left = generated[:, :, :-1, :, :]
    generated_x_griednt = generated_x_shift_right - generated_x_shift_left

    difference_x = torch.abs(true_x_gradient - generated_x_griednt)
    # loss_x_gradient = (torch.sum(difference_x ** 2)) / all
    loss_x_gradient = (torch.sum(difference_x)) / all

    true_y_shifted_right = real_image[:, :, :, 1:, :]
    true_y_shifted_left = real_image[:, :, :, :-1, :]
    true_y_gradient = true_y_shifted_right - true_y_shifted_left

    generated_y_shift_right = generated[:, :, :, 1:, ]
    generated_y_shift_left = generated[:, :, :, :-1:, :]
    generated_y_griednt = generated_y_shift_right - generated_y_shift_left

    difference_y = torch.abs(true_y_gradient - generated_y_griednt)
    # loss_y_gradient = (torch.sum(difference_y ** 2)) / all
    loss_y_gradient = (torch.sum(difference_y)) / all

    true_z_shifted_right = real_image[:, :, :, :, 1:]
    true_z_shifted_left = real_image[:, :, :, :, :-1]
    true_z_gradient = true_z_shifted_right - true_z_shifted_left

    generated_z_shift_right = generated[:, :, :, :, 1:]
    generated_z_shift_left = generated[:, :, :, :, :-1]
    generated_z_griednt = generated_z_shift_right - generated_z_shift_left

    difference_z = torch.abs(true_z_gradient - generated_z_griednt)
    # loss_z_gradient = (torch.sum(difference_z ** 2)) / all
    loss_z_gradient = (torch.sum(difference_z)) / all

    igdl = (loss_x_gradient + loss_y_gradient + loss_z_gradient)

    return igdl


def RLoss(generated, real_image):  # b x c x h x w
    b, c, h, w, d = generated.shape
    all = b * c * h * w * d

    true_x_shifted_right = real_image[:, :, 1:, :, :]
    true_x_shifted_left = real_image[:, :, :-1, :, :]
    true_x_gradient = true_x_shifted_left - true_x_shifted_right  # torch.abs

    generated_x_shift_right = generated[:, :, 1:, :, :]
    generated_x_shift_left = generated[:, :, :-1, :, :]
    generated_x_griednt = generated_x_shift_left - generated_x_shift_right  # torch.abs

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x ** 2)) / all

    true_y_shifted_right = real_image[:, :, :, 1:, :]
    true_y_shifted_left = real_image[:, :, :, :-1, :]
    true_y_gradient = true_y_shifted_left - true_y_shifted_right  # torch.abs

    generated_y_shift_right = generated[:, :, :, 1:, ]
    generated_y_shift_left = generated[:, :, :, :-1:, :]
    generated_y_griednt = generated_y_shift_left - generated_y_shift_right  # torch.abs

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y ** 2)) / all

    true_z_shifted_right = real_image[:, :, :, :, 1:]
    true_z_shifted_left = real_image[:, :, :, :, :-1]
    true_z_gradient = true_z_shifted_left - true_z_shifted_right  # torch.abs

    generated_z_shift_right = generated[:, :, :, :, 1:]
    generated_z_shift_left = generated[:, :, :, :, :-1]
    generated_z_graiednt = generated_z_shift_left - generated_z_shift_right  # torch.abs

    difference_z = true_z_gradient - generated_z_graiednt
    loss_z_gradient = (torch.sum(difference_z ** 2)) / all

    loss = loss_x_gradient + loss_y_gradient + loss_z_gradient

    return loss


class G_operator2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad1 = nn.ReplicationPad3d((0, 0, 0, 0, 1, 0))
        self.pad2 = nn.ReplicationPad3d((0, 0, 1, 0, 0, 0))
        self.pad3 = nn.ReplicationPad3d((1, 0, 0, 0, 0, 0))

    def forward(self, generated, feartures):
        generated_x_shift_right = generated[:, :, 1:, :, :]
        generated_x_shift_left = generated[:, :, :-1, :, :]
        generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)
        gx = self.pad1(generated_x_griednt)

        generated_y_shift_right = generated[:, :, :, 1:, ]
        generated_y_shift_left = generated[:, :, :, :-1:, :]
        generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)
        gy = self.pad2(generated_y_griednt)

        generated_z_shift_right = generated[:, :, :, :, 1:]
        generated_z_shift_left = generated[:, :, :, :, :-1]
        generated_z_griednt = torch.abs(generated_z_shift_left - generated_z_shift_right)
        gz = self.pad3(generated_z_griednt)

        g = gx + gy + gz
        out = torch.cat([g, feartures], dim=1)

        return out
