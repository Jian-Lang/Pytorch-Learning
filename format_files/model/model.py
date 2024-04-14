"""
@author: Lobster
@software: PyCharm
@file: model.py
@time: 2024/4/9 16:34
"""
import torch
from torch import nn


class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.linear_1 = nn.Linear(2, 1)

        self.sigmoid = nn.Sigmoid()

        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, height, weigh):

        height = height.unsqueeze(1)

        weigh = weigh.unsqueeze(1)

        height = self.batch_norm(height)

        weigh = self.batch_norm(weigh)

        x = torch.cat((height, weigh), dim=1)

        x = self.linear_1(x)

        x = self.sigmoid(x)

        return x

