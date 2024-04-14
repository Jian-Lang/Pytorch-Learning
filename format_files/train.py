import logging
import os
import sys
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from dataloader.dataset import MyData, custom_collate_fn
from model.model import Model
import random
from functools import partial

from utils.parsers import build_parser

BLUE = '\033[94m'
ENDC = '\033[0m'


def seed_init(seed):
    seed = int(seed)

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):

    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")

    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")

    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")

    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")

    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")

    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.metric}_{timestamp}"

    father_folder_name = args.output_path

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    os.mkdir(folder_path)

    logger = logging.getLogger()

    logger.handlers = []

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}")

    for i in range(len(model_name_list)):

        if model_name_list[i] != f'checkpoint_{min_turn}_epoch.pkl' and model_name_list[i] != 'log.txt':

            os.remove(os.path.join(f'{father_folder_name}/{folder_name}', model_name_list[i]))


def force_stop(msg):
    print(msg)

    sys.exit(1)


def delete_special_tokens(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def train_val(args):

    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)

    device = torch.device(args.device)

    train_data = MyData(os.path.join(args.dataset_path, 'train.pkl'))

    valid_data = MyData(os.path.join(args.dataset_path, 'valid.pkl'))

    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    model = Model()

    model = model.to(device)

    loss_fn = torch.nn.BCELoss()

    loss_fn.to(device)

    if args.optim == 'Adam':

        optim = Adam(model.parameters(), args.lr)

    elif args.optim == 'SGD':

        optim = SGD(model.parameters(), args.lr)

    else:

        force_stop('Invalid parameter optim!')

    min_total_valid_loss = 1008611

    min_turn = 0

    init_turn = 0

    if args.load_checkpoint:

        # 如果加载checkpoint，那么就从checkpoint中加载模型和优化器的状态

        # 同时，先对模型进行评估，得到最小的valid loss(默认从检查点中加载的模型是最优的)

        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        init_turn = checkpoint['epoch']

        min_turn = init_turn

        min_total_valid_loss = 0

        with torch.no_grad():

            for batch in tqdm(valid_data_loader, desc='Validating Progress'):

                batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

                height, weigh, label = batch

                output = model.forward(height, weigh)

                label = label.unsqueeze(1)

                loss = loss_fn(output, label)

                min_total_valid_loss += loss

        logger.info(f"Load checkpoint from {args.checkpoint_path} successfully!")

        logger.info(f"Start from {init_turn + 1} epoch!")

    print_init_msg(logger, args)

    for i in range(args.epochs - init_turn):

        logger.info(f"-----------------------------------Epoch {i + 1 + init_turn} Start!-----------------------------------")

        min_train_loss, total_valid_loss = run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader,
                                                         device)

        logger.info(f"[ Epoch {i + 1 + init_turn} (train) ]: loss = {min_train_loss}")

        logger.info(f"[ Epoch {i + 1 + init_turn} (valid) ]: total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:

            min_total_valid_loss = total_valid_loss

            min_turn = i + 1 + init_turn

        logger.critical(
            f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")

        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optim.state_dict(),
                      "epoch": i + 1,}

        if not os.path.exists(args.output_path):

            os.makedirs(args.output_path)

        path_checkpoint = f"{father_folder_name}/{folder_name}/checkpoint_{i + 1 + init_turn}_epoch.pkl"

        torch.save(checkpoint, path_checkpoint)

        logger.info("Model has been saved successfully!")

        delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")

        if (i + 1) - min_turn > args.early_stop_turns:
            break

    delete_model(father_folder_name, folder_name, min_turn)

    logger.info(BLUE + "Training is ended!" + ENDC)

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")


def run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader, device):

    model.train()

    min_train_loss = 1008611

    for batch in tqdm(train_data_loader, desc='Training Progress'):

        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

        height,weigh, label = batch

        output = model.forward(height,weigh)

        label = label.unsqueeze(1)

        loss = loss_fn(output, label)

        optim.zero_grad()

        loss.backward()

        optim.step()

        if min_train_loss > loss:
            min_train_loss = loss

    model.eval()

    total_valid_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_data_loader, desc='Validating Progress'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            height, weigh, label = batch

            output = model.forward(height, weigh)

            label = label.unsqueeze(1)

            loss = loss_fn(output, label)

            total_valid_loss += loss

    return min_train_loss, total_valid_loss


def main():
    parser = build_parser('train')

    args = parser.parse_args()

    # 设置随机种子

    seed_init(args.seed)

    train_val(args)


if __name__ == '__main__':
    main()
