from abc import ABC
from collections import OrderedDict

from torch import nn

act_fc = nn.ReLU()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)

class CNNModel(nn.Module, ABC):
    def __init__(self, output_size):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(1),

            nn.Conv1d(1, 2, 16),
            nn.BatchNorm1d(2),
            nn.ReLU(),

            nn.Conv1d(2, 4, 8),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Conv1d(4, 8, 6),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, 4),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),


        )

        self.se = SELayer(channel=64)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 64)),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = x.view((x.size()[0], 1, x.size()[1]))
        x = self.conv(x)
        x = self.se(x)
        x = self.fc(x)
        x = x.view(x.size()[0], x.size()[2])
        return x
