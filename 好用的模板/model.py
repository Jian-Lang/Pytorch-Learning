"""
@author: Lobster
@software: PyCharm
@file: model.py
@time: 2023/9/27 20:51
"""
import torch.nn


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size,1),
            torch.nn.ReLU()
        )

    def forward(self, x):

        return self.model(x)