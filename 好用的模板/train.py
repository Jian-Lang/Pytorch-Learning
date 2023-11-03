"""
@author: Lobster
@software: PyCharm
@file: train.py
@time: 2023/10/13 15:12
"""
import logging
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from dataset import MyData, custom_collate_fn
from model import Model


# 文本颜色设置

BLUE = '\033[94m'
ENDC = '\033[0m'


def make_saving_folder_and_logger(args):

    # 创建文件夹和日志文件，用于记录训练结果和模型

    # 获取当前时间戳

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件夹名称

    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.metric}_{timestamp}"

    # 指定文件夹的完整路径

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):

        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    # 创建文件夹

    os.mkdir(folder_path)

    os.mkdir(os.path.join(folder_path, "trained_model"))

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

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn):

    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")

    for i in range(len(model_name_list)):

        if model_name_list[i] != f'model_{min_turn}.pth':

            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))


def force_stop(msg):

    print(msg)

    sys.exit(1)


def train(args):

    # 通过args解析出所有参数

    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)

    # device

    device = torch.device(args.device)

    # 加载数据集

    train_data = MyData(os.path.join(args.dataset_path, args.dataset_id, 'train.pickle'))

    valid_data = MyData(os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'valid.pickle')))

    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    # 加载模型

    model = Model(input_size=24)

    model = model.to(device)

    # 定义损失函数

    if args.loss == 'BCE':

        loss_fn = torch.nn.BCELoss()

    elif args.loss == 'MSE':

        loss_fn = torch.nn.MSELoss()

    else:

        force_stop('Invalid parameter loss!')

    loss_fn.to(device)

    # 定义优化器

    if args.optim == 'Adam':

        optim = Adam(model.parameters(), args.lr)

    elif args.optim == 'SGD':

        optim = SGD(model.parameters(), args.lr)

    else:

        force_stop('Invalid parameter optim!')

    # 定义学习率衰减

    decayRate = args.decay_rate

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decayRate)

    # 定义训练过程的一些参数

    min_total_valid_loss = 1008611

    min_turn = 0

    # 开始训练

    logger.info(BLUE + 'Device: ' + ENDC + f"{device} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")

    logger.info(BLUE + "Learning Decay: " + ENDC + f"{args.decay_rate}")

    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")

    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")

    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")

    logger.info(BLUE + "Training Starts!" + ENDC)

    for i in range(args.epochs):

        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")

        min_train_loss, total_valid_loss = run_one_epoch(model, loss_fn, optim,train_data_loader, valid_data_loader,device)

        scheduler.step()

        logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")

        logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:

            min_total_valid_loss = total_valid_loss

            min_turn = i + 1

        logger.critical(f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")

        torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth")

        logger.info("Model has been saved successfully!")

        if (i + 1) - min_turn > args.early_stop_turns:

            break

    delete_model(father_folder_name, folder_name, min_turn)

    logger.info(BLUE + "Training is ended!" + ENDC)


def run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader, device):

    # 训练部分

    model.train()

    min_train_loss = 1008611

    for batch in tqdm(train_data_loader, desc='Training Progress'):

        aggregated_amounts, target = batch

        aggregated_amounts = aggregated_amounts.to(device)

        target = target.to(device)

        target = target.type(torch.float32)

        output = model.forward(aggregated_amounts)

        loss = loss_fn(output, target)

        # 通过损失，优化参数

        optim.zero_grad()

        loss.backward()

        optim.step()

        if min_train_loss > loss:

            min_train_loss = loss

    # 验证环节

    model.eval()

    total_valid_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_data_loader, desc='Validating Progress'):

            aggregated_amounts, target = batch

            aggregated_amounts = aggregated_amounts.to(device)

            target = target.to(device)

            target = target.type(torch.float32)

            output = model.forward(aggregated_amounts)

            loss = loss_fn(output, target)

            total_valid_loss += loss

    return min_train_loss,total_valid_loss


# 主函数，所有训练参数在这里调整

def main():

    # 创建一个ArgumentParser对象

    parser = argparse.ArgumentParser()

    # 运行前命令行参数设置

    parser.add_argument('--device', default='cuda:0', type=str, help='device used in training')

    parser.add_argument('--metric', default='MSE', type=str, help='the judgement of the training')

    parser.add_argument('--save', default='train_results', type=str, help='folder to save the results')

    parser.add_argument('--epochs', default=10, type=int, help='max number of training epochs')

    # 注意，大的batch_size，梯度比较平滑，可以设置大的learning rate，vice visa

    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')

    parser.add_argument('--early_stop_turns', default=3, type=int, help='early stop turns of training')

    parser.add_argument('--loss', default='MSE', type=str, help='loss function, options: BCE, MSE')

    parser.add_argument('--optim', default='Adam', type=str, help='optim, options: SGD, Adam')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    # trick: 先按1.0，即不decay学习率训练，震荡不收敛，可以适当下调

    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')

    parser.add_argument('--dataset_id', default='gofundme', type=str, help='id of dataset')

    parser.add_argument('--dataset_path', default='data', type=str, help='path of dataset')

    parser.add_argument('--model_id', default='mlp', type=str, help='id of model')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':

    main()





