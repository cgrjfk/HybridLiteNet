import torch
import torch.nn as nn
import torch.nn.functional as F
from SwishActivate import Swish


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcite, self).__init__()
        self.se_reduce = nn.Conv2d(in_channels, reduced_dim, kernel_size=1)
        self.se_expand = nn.Conv2d(reduced_dim, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = torch.relu(self.se_reduce(scale))
        scale = torch.sigmoid(self.se_expand(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio):
        super(MBConv, self).__init__()
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio
        self.stride = stride
        self.has_residual = (in_channels == out_channels) and (stride == 1)

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(mid_channels)

        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.se = SqueezeExcite(mid_channels, int(in_channels * se_ratio))
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.swish = Swish()

    def forward(self, x):
        if self.expand_ratio != 1:
            out = self.swish(self.bn0(self.expand_conv(x)))
        else:
            out = x
        out = self.swish(self.bn1(self.depthwise_conv(out)))
        out = self.se(out)
        out = self.bn2(self.project_conv(out))

        if self.has_residual:
            out = out + x
        return out
