"""
@author: Lobster
@software: PyCharm
@file: dataset.py
@time: 2023/9/27 20:51
"""
import torch.utils.data
import pandas as pd


def custom_collate_fn(batch):

    aggregated_amounts,target = zip(*batch)

    # 返回一个包含三个张量的元组

    return torch.log(torch.tensor(aggregated_amounts) + 1), torch.tensor(target).unsqueeze(-1)


class MyData(torch.utils.data.Dataset):

    def __init__(self,path):
        super().__init__()
        self.path = path
        self.dataframe = pd.read_pickle(path)
        self.aggregated_amounts_list = self.dataframe['aggregated_amounts']
        self.target_list = self.dataframe['target_re']

    def __getitem__(self, item):

        aggregated_amounts = self.aggregated_amounts_list[item]

        target = self.target_list[item]

        return aggregated_amounts,target

    def __len__(self):
        return len(self.dataframe)