import torch
from torch import nn

from torch.nn.parameter import Parameter


class EfficientChannelAttention(nn.Module):

    def __init__(self, channel, kernel_size=2):
        super(EfficientChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        y = self.sigmoid(y)
        return x * y.expand_as(x)


