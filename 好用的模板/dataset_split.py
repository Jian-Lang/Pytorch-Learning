"""
@author: Lobster
@software: PyCharm
@file: dataset_split.py
@time: 2023/10/9 8:16
"""
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    initial_data_frame = pd.read_pickle(r'')

    # 下面的部分, 编写对原始数据集dataframe的各种修改即可

    # 生成训练集、测试集和验证集

    train_data_frame, others_data_frame = train_test_split(initial_data_frame, test_size=0.2, random_state=9)

    test_data_frame, valid_data_frame = train_test_split(others_data_frame, test_size=0.5, random_state=18)

    # 修正索引

    train_data_frame.reset_index(drop=True, inplace=True)

    test_data_frame.reset_index(drop=True, inplace=True)

    valid_data_frame.reset_index(drop=True, inplace=True)

    # 保存为pickle文件

    train_data_frame.to_pickle(r'')

    test_data_frame.to_pickle(r'')

    valid_data_frame.to_pickle(r'')



