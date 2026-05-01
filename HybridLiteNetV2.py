import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# ECA Attention
# ======================
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


# ======================
# ConvNeXt Block
# ======================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        return x + identity


# ======================
# Linear Attention
# ======================
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def phi(self, x):
        return F.elu(x) + 1  

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv

        q = self.phi(q)
        k = self.phi(k)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(2)) + 1e-6)

        out = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)
        out = out.reshape(B, N, C)

        return self.proj(out)


# ======================
# Transformer Block
# ======================
class TransBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ======================
# 主模型
# ======================
class HybridLiteNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )

        # ---- Stage1 ----
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(48),
            ConvNeXtBlock(48),
            ECA(48)
        )

        # ---- Downsample ----
        self.down1 = nn.Conv2d(48, 96, 3, stride=2, padding=1)

        # ---- Stage2----
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(96),
            ConvNeXtBlock(96)
        )

        self.trans_dim = 96
        self.trans = TransBlock(self.trans_dim, heads=4)

        # ---- Downsample ----
        self.down2 = nn.Conv2d(96, 160, 3, stride=2, padding=1)

        # ---- Stage3 ----
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(160),
            ConvNeXtBlock(160),
            ECA(160)
        )

        # ---- Head----
        self.head = nn.Sequential(
            nn.Conv2d(160, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.Conv2d(256, 320, 1),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.down1(x)

        x = self.stage2(x)

        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.trans(t)
        x = t.transpose(1, 2).reshape(B, C, H, W)

        x = self.down2(x)
        x = self.stage3(x)

        x = self.head(x)
        return x



if __name__ == "__main__":
    model = HybridLiteNetV2()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output:", y.shape)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {total_params:.3f}M")
