import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channelFilter = ChannelFilter(in_channel, reduction_ratio)
        self.spatialFilter = SpatialFilter()

    def forward(self, x):
        x_out = self.channelFilter(x)
        x_out = self.spatialFilter(x_out)

        return x_out


class ChannelFilter(nn.Module):
    def __init__(self, in_channel, reduction_ratio):
        super(ChannelFilter, self).__init__()
        self.in_channel = in_channel
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channel, in_channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channel // reduction_ratio, in_channel)
        )
        

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(2).unsqueeze(3).expand_as(x)

        return channel_attention * x


class SpatialFilter(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialFilter, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.batch_norm = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 通道滤波
        x_filtered = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_filtered = self.relu(self.batch_norm(self.conv(x_filtered)))
        spatial_attention = torch.sigmoid(x_filtered)

        return spatial_attention * x
