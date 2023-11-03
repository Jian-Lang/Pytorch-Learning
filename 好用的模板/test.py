"""
@author: Lobster
@software: PyCharm
@file: test.py
@time: 2023/9/27 20:51
"""
import argparse
import os
from datetime import datetime
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,mean_squared_error
from dataset import MyData,custom_collate_fn
from model import Model

# 文本颜色设置

BLUE = '\033[94m'
ENDC = '\033[0m'


def test(args):

    # device

    device = torch.device(args.device)

    # model id

    model_id = args.model_id

    # dataset id

    dataset_id = args.dataset_id

    # metric

    metric = args.metric

    # 创建文件夹和日志文件，用于记录验证集的结果

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件夹名称
    folder_name = f"test_{model_id}_{dataset_id}_{metric}_{timestamp}"

    # 指定文件夹的完整路径

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):

        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    # 创建文件夹
    os.mkdir(folder_path)

    # 配置日志记录

    # 创建logger对象

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # 创建控制台处理器

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    # 创建文件处理器

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    # 设置日志格式

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    # 将处理器添加到logger对象中

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    # 加载数据集

    batch_size = args.batch_size

    test_data = MyData(os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'test.pickle')))

    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=custom_collate_fn)

    # 加载训练好的模型, 这里记得import你的模型

    model = torch.load(args.model_path)

    # 定义验证相关参数

    total_test_step = 0

    total_test_loss = 0

    # 开始验证

    logger.info(BLUE + 'Device: ' + ENDC + f"{device} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Training Starts!" + ENDC)

    model.eval()

    with torch.no_grad():

        for batch in tqdm(test_data_loader, desc='Testing'):

            input_, target = batch

            input_ = input_.to(device)

            target = target.to(device)

            target = target.view(-1, 1)

            output = model(input_)

            output = output.to('cpu')

            target = target.to('cpu')

            if metric == 'RMSE':

                loss = mean_squared_error(target,output)

            elif metric == 'AUC':

                loss = roc_auc_score(output, target)

            else:

                print('Invalid Metric!')

            total_test_step += 1

            total_test_loss += loss

    logger.warning(f"[ Test Result ]:  {metric} = {total_test_loss / total_test_step}")

    logger.info("Test is ended!")


def main():
    
    # 创建一个ArgumentParser对象

    parser = argparse.ArgumentParser()

    # 运行前命令行参数设置

    parser.add_argument('--device', default='cuda:0', type=str, help='device used in testing')

    parser.add_argument('--metric', default='RMSE', type=str, help='the judgement of the training')

    parser.add_argument('--save', default='test_results', type=str, help='folder to save the results')

    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')

    parser.add_argument('--dataset_id', default='gofundme', type=str, help='id of dataset')

    parser.add_argument('--dataset_path', default='data', type=str, help='path of dataset folder')

    parser.add_argument('--model_id', default='mlp', type=str, help='id of model')

    parser.add_argument('--model_path',
                        default=r'D:\MachineLearningProject\mlp_with_new_format_files\train_results\train_mlp_gofundme_MSE_2023-10-14_15-38-18\trained_model\model_4.pth',
                        type=str, help='path of trained model')

    args = parser.parse_args()

    test(args)


if __name__ == "__main__":
    
    main()
    

