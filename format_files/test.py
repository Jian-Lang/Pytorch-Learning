import argparse
import os
from datetime import datetime
import logging
from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataloader.dataset import MyData, custom_collate_fn
import random
import numpy as np
from scipy.stats import spearmanr

from utils.parsers import build_parser

from model.model import Model

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

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_path} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Testing Starts!" + ENDC)


def delete_special_tokens(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:

        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:

        file.write(content)


def test(args):

    device = torch.device(args.device)

    model_id = args.model_id

    dataset_id = args.dataset_id

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"test_{model_id}_{dataset_id}_{timestamp}"

    father_folder_name = args.save

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

    batch_size = args.batch_size

    test_data = MyData(os.path.join(os.path.join(args.dataset_path, 'test.pkl')))

    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=custom_collate_fn)

    model = Model()

    model.to(device)

    checkpoint = torch.load(args.model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    total_test_step = 0

    print_init_msg(logger, args)

    model.eval()

    with torch.no_grad():

        for batch in tqdm(test_data_loader, desc='Testing'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            height, weigh, label = batch

            output = model.forward(height, weigh)

            label = label.unsqueeze(1)

            output = output.to('cpu')

            label = label.to('cpu')

            output = np.array(output)

            output = [1 if i > 0.5 else 0 for i in output]

            label = np.array(label)

            f1 = f1_score(label, output)

            total_test_step += 1

    logger.warning(f"[ Test Result ]: {args.metric[0]} = {f1}\n")

    logger.info("Test is ended!")

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")


def main():

    parser = build_parser('test')

    args = parser.parse_args()

    # 设置随机种子

    seed_init(args.seed)

    test(args)


if __name__ == "__main__":

    main()
