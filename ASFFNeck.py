import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    def __init__(self, in_channels_list):
        super(ASFF, self).__init__()
        self.in_channels_list = in_channels_list
        self.num_channels = len(in_channels_list)

        self.adaptive_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            for in_channels in in_channels_list
        ])

        self.channel_enhance_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            for in_channels in in_channels_list
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels_list), sum(in_channels_list), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(sum(in_channels_list)),
            nn.ReLU(inplace=True),
            nn.Conv2d(sum(in_channels_list), sum(in_channels_list), kernel_size=1, stride=1, padding=0)
        )

    def forward(self, feature_maps):
        assert len(feature_maps) == self.num_channels

        weights = [F.softmax(adaptive_conv(fm), dim=1) for adaptive_conv, fm in zip(self.adaptive_convs, feature_maps)]

        enhanced_features = [channel_enhance_conv(fm) for channel_enhance_conv, fm in
                             zip(self.channel_enhance_convs, feature_maps)]

        weighted_features = [weight * fm for weight, fm in zip(weights, enhanced_features)]
        fused_features = torch.cat(weighted_features, dim=1)
        fused_features = self.fusion_conv(fused_features)

        return fused_features
