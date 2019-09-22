from collections import OrderedDict
import torch
from torch import nn


class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class Concat(nn.Module):
    def __init__(self, dim=0):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *seq):
        return torch.cat(seq, dim=self.dim)


class Block(nn.Module):
    """Pre-activated ResNet block.
    """

    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width, affine=False)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.add = EltwiseAdd()

    def forward(self, x):
        h = self.conv0(self.relu0(self.bn0(x)))
        h = self.conv1(self.relu1(self.bn1(h)))
        return self.add(x, h)


class DownsampleBlock(nn.Module):
    """Downsample block.

    Does F.avg_pool2d + torch.cat instead of strided conv.
    """

    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(width // 2, width, 3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.pool = nn.AvgPool2d(3, padding=1, stride=2)
        self.concat = Concat(dim=1)
        self.add = EltwiseAdd()

    def forward(self, x):
        h = self.conv0(self.relu0(self.bn0(x)))
        h = self.conv1(self.relu1(self.bn1(h)))
        x_d = self.pool(x)
        x_d = self.concat(x_d, torch.zeros_like(x_d))
        return self.add(x_d, h)


class WRN_McDonnell_Eval(nn.Module):
    """Implementation of modified Wide Residual Network.

    Differences with pre-activated ResNet and Wide ResNet:
        * BatchNorm has no affine weight and bias parameters
        * First layer has 16 * width channels
        * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
        * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv

    First and last convolutional layers are kept in float32.
    """

    def __init__(self, depth, width, num_classes):
        super().__init__()
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6

        self.conv0 = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)

        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)

        self.bn = nn.BatchNorm2d(widths[2], affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_last = nn.Conv2d(widths[2], num_classes, 1, bias=False)
        self.bn_last = nn.BatchNorm2d(num_classes, affine=False)
        self.pool = nn.AvgPool2d(8)

    def _make_block(self, width, n, downsample=False):
        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width)
            return Block(width)
        return nn.Sequential(OrderedDict(('block%d' % i, select_block(i))
                                         for i in range(n)))

    def forward(self, x):
        h = self.conv0(x)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        h = self.relu(self.bn(h))
        h = self.conv_last(h)
        h = self.bn_last(h)
        h = self.pool(h).view(h.shape[0], -1)
        return h
