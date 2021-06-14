import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from models.ConvBlockAttention import ConvBlockAttention
from models.EfficientChannelAttention import EfficientChannelAttention
from models.SEAttention import SEAttention


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=2, downsample=None, attention=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)  # inplace=True 节省内存

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.attention = attention(planes, 2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out_size = out.size(2)

        residual_size = residual.size(2)
        is_large = residual_size - out_size

        zero_padding = torch.zeros(
            abs(residual_size - out_size) * residual.size(1) * residual.size(0)
        ).view(residual.size(0), residual.size(1), -1)

        if is_large:
            out = torch.cat([out, zero_padding], dim=2)
        else:
            residual = torch.cat([residual, zero_padding], dim=2)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, attention=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm1d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.cbam = None
        if attention:
            self.cbam = ConvBlockAttention(planes * 4, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, attention):
        super(ResNet, self).__init__()
        self.in_planes = 2

        self.conv1 = nn.Conv1d(1, 2, kernel_size=16, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention
        self.layers = nn.Sequential(
            self.make_layer(block, 4, layers[0], kernel_size=4),
            self.make_layer(block, 8, layers[1], kernel_size=4),
            self.make_layer(block, 32, layers[2], kernel_size=2),
            self.make_layer(block, 64, layers[3], kernel_size=2),
            self.make_layer(block, 128, layers[4], kernel_size=2)
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 128))

        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, block, planes, blocks, kernel_size=2, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = [block(self.in_planes, planes, stride, kernel_size, downsample, attention=self.attention)]

        self.in_planes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, kernel_size=kernel_size, attention=self.attention))

        return nn.Sequential(*layers)

    def shortcut(self, x, feature):
        downsample = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(128)
        )

        residual = downsample(x)

        feature_size = feature.size(2)
        residual_size = residual.size(2)
        is_large = residual_size - feature_size

        zero_padding = torch.zeros(
            abs(residual_size - feature_size) * residual.size(1) * residual.size(0)
        ).view(residual.size(0), residual.size(1), -1)

        if is_large:
            feature = torch.cat([feature, zero_padding], dim=2)
        else:
            residual = torch.cat([residual, zero_padding], dim=2)

        return feature + residual

    def feature_extract(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        # residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        # x = self.shortcut(residual, x)

        x = self.avg_pooling(x)

        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)


def attention_resnet(num_classes, attention=ConvBlockAttention):
    return ResNet(BasicBlock, [2, 2, 2, 2, 2], num_classes=num_classes, attention=attention)



