"""Portable inference network for the CST CIFAR-10 residual CNN."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def conv_bn(inputs, outputs, stride=1, kernel=3):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(outputs),
    )


class Block(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.first = conv_bn(inputs, outputs, stride)
        self.second = conv_bn(outputs, outputs)
        self.skip = nn.Identity() if stride == 1 else conv_bn(inputs, outputs, stride, kernel=1)

    def forward(self, x):
        return F.relu(self.second(F.relu(self.first(x))) + self.skip(x))


class Classifier(nn.Module):
    def __init__(self, *, middle_gain, first_gain=None):
        super().__init__()
        self.trunk = nn.Sequential(
            conv_bn(3, 32), nn.ReLU(), Block(32, 32, 1), Block(32, 64, 2),
            Block(64, 96, 2), Block(96, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.stem = nn.Linear(128, 2048)
        self.norm = nn.LayerNorm(2048)
        self.recurrent = nn.Linear(2048, 2048, bias=False)
        self.bias = nn.Parameter(torch.zeros(2048))
        self.readout = nn.Linear(2048, 10)
        self.middle_gain = middle_gain
        self.first_gain = first_gain
        self.trunk.to(memory_format=torch.channels_last)

    def forward(self, x):
        context = torch.autocast("cuda", dtype=torch.bfloat16) if x.is_cuda else nullcontext()
        with context:
            pooled = self.trunk(x.contiguous(memory_format=torch.channels_last))
        pooled = pooled.float()
        if self.first_gain is None:
            stem = self.stem(pooled)
        else:
            stem = F.linear(pooled, self.stem.weight) * self.first_gain + self.stem.bias
        h = F.gelu(self.norm(stem))
        return self.readout(F.gelu(self.recurrent(h) * self.middle_gain + self.bias))
