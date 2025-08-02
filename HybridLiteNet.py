import torch
import torch.nn as nn
import torch.nn.functional as F

from AttentionBlock.MBConvBolck import MBConv
from ActFunction.MishActivate import Mish
from ActFunction.SwishActivate import Swish
from AttentionBlock.SEBolck import SEBlock
from runs.ASFFNeck import ASFF


# ---- RMSNorm and TransBlock  ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x * (self.scale / (norm / (x.shape[-1] ** 0.5 + self.eps)))


class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def phi(self, x):  # 高斯核映射
        return torch.exp(-0.5 * x.pow(2))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        q, k = self.phi(q), self.phi(k)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(2)) + 1e-6)
        out = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z).reshape(B, N, C)
        return self.proj(out)


class DepthwiseMLP(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.conv1 = nn.Conv1d(dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv2(self.act(self.conv1(x)))
        return x.transpose(1, 2)


class TransBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = PerformerAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = DepthwiseMLP(dim, mlp_ratio)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma * self.attn(self.norm1(x))
        x = x + self.gamma * self.mlp(self.norm2(x))
        return x


# ---- HybridLiteNet ----
class HybridLiteNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.swish = Swish()

        def create_block(in_channels, out_channels, kernel_size, dilation, activation, attention):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                activation,
                attention(out_channels)
            )

        # 多尺度分支
        self.conv1x1 = create_block(3, 32, 1, 1, Mish(), SEBlock)
        self.conv3x3 = create_block(3, 32, 3, 2, Swish(), SEBlock)
        self.conv5x5 = create_block(3, 32, 5, 2, Mish(), SEBlock)

        self.asff = ASFF([32, 32, 32])

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.swish
        )

        # MBConv Backbone
        self.blocks = nn.Sequential(
            MBConv(64, 32, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.25),
            MBConv(32, 32, expand_ratio=4, kernel_size=3, stride=2, se_ratio=0.25),
            MBConv(32, 48, expand_ratio=4, kernel_size=5, stride=2, se_ratio=0.25),
            MBConv(48, 96, expand_ratio=4, kernel_size=3, stride=2, se_ratio=0.25),
            MBConv(96, 128, expand_ratio=4, kernel_size=5, stride=1, se_ratio=0.25),
            MBConv(128, 192, expand_ratio=4, kernel_size=5, stride=1, se_ratio=0.25)
        )

        # fusion
        self.fusion_feature = nn.Sequential(
            nn.Conv2d(288, 160, 1),  # 96+192=288
            nn.BatchNorm2d(160),
            Swish(),
            nn.Dropout2d(0.4)
        )

        # light Transformer block
        self.attn_dim = 160
        self.trans_block = TransBlock(self.attn_dim, heads=4, mlp_ratio=2)

        # 分类头
        self.head = nn.Sequential(
            nn.Conv2d(160, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            self.swish,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x0 = self.conv1x1(x)
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)

        max_h = max(x0.size(2), x1.size(2), x2.size(2))
        max_w = max(x0.size(3), x1.size(3), x2.size(3))
        x0 = F.interpolate(x0, size=(max_h, max_w), mode='bilinear', align_corners=False)
        x1 = F.interpolate(x1, size=(max_h, max_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(max_h, max_w), mode='bilinear', align_corners=False)

        feature_maps = self.asff([x0, x1, x2])
        stem_out = self.stem(feature_maps)
        backbone_out = self.blocks(stem_out)

        feature_maps = F.interpolate(feature_maps, size=backbone_out.shape[2:], mode='bilinear', align_corners=False)
        residual = torch.cat((feature_maps, backbone_out), dim=1)
        features = self.fusion_feature(residual)

        B, C, H, W = features.shape
        t = features.flatten(2).transpose(1, 2)
        t = self.trans_block(t)
        features = t.transpose(1, 2).reshape(B, C, H, W)

        out = self.head(features)
        return out
