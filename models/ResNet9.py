import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, c_in, c_out):
        super(Residual, self).__init__()
        self.pre = ConvBN(c_in, c_out)
        self.conv_bn1 = ConvBN(c_out, c_out)
        self.conv_bn2 = ConvBN(c_out, c_out)

    def forward(self, x):
        x = self.pre(x)
        x = F.max_pool2d(x, 2)
        return self.conv_bn2(self.conv_bn1(x)) + x


class ResNet9(nn.Module):
    def __init__(self, d1, d2, d3, d4):
        super(ResNet9, self).__init__()
        self.pre = ConvBN(3, d1)
        self.residual1 = Residual(d1, d2)
        self.inter = ConvBN(d2, d3)
        self.residual2 = Residual(d3, d4)
        self.linear = nn.Linear(d4, 100, bias=False)

    def forward(self, x):
        x = self.pre(x)
        x = self.residual1(x)
        x = self.inter(x)
        x = F.max_pool2d(x, 2)
        x = self.residual2(x)
        x = F.max_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x * 0.125
