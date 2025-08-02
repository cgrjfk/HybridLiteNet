import torch
import torch.nn as nn
import torch.nn.functional as F


# Mish 激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
