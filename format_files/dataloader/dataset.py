import numpy as np
import torch.utils.data
import pandas as pd
from functools import partial


def custom_collate_fn(batch):
    height, weigh, label = zip(*batch)

    return torch.tensor(height,dtype=torch.float), torch.tensor(weigh,dtype=torch.float), torch.tensor(label,dtype=torch.float)


class MyData(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()

        self.path = path

        self.dataframe = pd.read_pickle(path)

        self.height_list = self.dataframe['Height'].tolist()

        self.weigh_list = self.dataframe['Weigh'].tolist()

        self.label_list = self.dataframe['Gender'].tolist()

    def __getitem__(self, index):

        height = self.height_list[index]

        weigh = self.weigh_list[index]

        label = self.label_list[index]

        return height, weigh, label

    def __len__(self):

        return len(self.dataframe)


if __name__ == "__main__":

    dataset = MyData(r'D:\MultiModalPopularityPrediction\data\MicroLens-100k\valid.pkl')

    custom_collate_fn_partial = partial(custom_collate_fn, num_of_retrieved_items=5, num_of_frames=3)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=custom_collate_fn_partial)

    for batch in dataloader:
        visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
            retrieved_textual_feature_embedding, retrieved_label, label = batch
