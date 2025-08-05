import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


'''
# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, dropout_prob=0.2):
        super(SEBlock, self).__init__()
        reduced_channels = in_channels // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.gelu(self.fc1(y))
        y = self.dropout(y)
        y = self.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)
'''

