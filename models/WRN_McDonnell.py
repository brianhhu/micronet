from collections import OrderedDict
import math
import torch
from torch import nn
import torch.nn.functional as F


def init_weight(*args):
    return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))


class ForwardSign(torch.autograd.Function):
    """Fake sign op for 1-bit weights.

    See eq. (1) in https://arxiv.org/abs/1802.08530

    Does He-init like forward, and nothing on backward.
    """

    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2. / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    @staticmethod
    def backward(ctx, g):
        return g


class ModuleBinarizable(nn.Module):

    def __init__(self, binarize=False):
        super().__init__()
        self.binarize = binarize

    def _get_weight(self, name):
        w = getattr(self, name)
        return ForwardSign.apply(w) if self.binarize else w

    def forward(self):
        pass


class Conv(ModuleBinarizable):
    """Binarized convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, binarize=False):
        super().__init__(binarize)
        self.padding = kernel_size // 2
        self.register_parameter('weight', init_weight(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        x = F.conv2d(x, self._get_weight('weight'), padding=self.padding)
        return x


class Block(ModuleBinarizable):
    """Pre-activated ResNet block.
    """

    def __init__(self, width, binarize=False):
        super().__init__(binarize)
        self.bn0 = nn.BatchNorm2d(width, affine=False, track_running_stats=False)
        self.register_parameter('conv0', init_weight(width, width, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False, track_running_stats=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self._get_weight('conv0'), padding=1)
        h = F.conv2d(F.relu(self.bn1(h)), self._get_weight('conv1'), padding=1)
        return x + h


class DownsampleBlock(ModuleBinarizable):
    """Downsample block.

    Does F.avg_pool2d + torch.cat instead of strided conv.
    """

    def __init__(self, width, binarize=False):
        super().__init__(binarize)
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False, track_running_stats=False)
        self.register_parameter('conv0', init_weight(width, width // 2, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False, track_running_stats=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self._get_weight('conv0'), padding=1, stride=2)
        h = F.conv2d(F.relu(self.bn1(h)), self._get_weight('conv1'), padding=1)
        x_d = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
        x_d = torch.cat([x_d, torch.zeros_like(x_d)], dim=1)
        return x_d + h


class WRN_McDonnell(ModuleBinarizable):
    """Implementation of modified Wide Residual Network.

    Differences with pre-activated ResNet and Wide ResNet:
        * BatchNorm has no affine weight and bias parameters
        * First layer has 16 * width channels
        * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
        * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv

    First and last convolutional layers are kept in float32.
    """

    def __init__(self, depth, width, num_classes, binarize=False):
        super().__init__()
        self.binarize = binarize
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6

        self.conv0 = Conv(3, widths[0], 3, binarize)

        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)

        self.bn = nn.BatchNorm2d(widths[2], affine=False, track_running_stats=False)
        self.conv_last = Conv(widths[2], num_classes, 1, binarize)
        self.bn_last = nn.BatchNorm2d(num_classes, affine=False, track_running_stats=False)

    def _make_block(self, width, n, downsample=False):
        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width, self.binarize)
            return Block(width, self.binarize)
        return nn.Sequential(OrderedDict(('block%d' % i, select_block(i))
                                         for i in range(n)))

    def forward(self, x):
        h = self.conv0(x)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        h = F.relu(self.bn(h))
        h = self.conv_last(h)
        h = self.bn_last(h)
        return F.avg_pool2d(h, kernel_size=h.shape[-2:]).view(h.shape[0], -1)
