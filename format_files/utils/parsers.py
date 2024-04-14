"""
@author: Lobster
@software: PyCharm
@file: parsers.py
@time: 2024/3/27 20:47
"""
import argparse
import yaml


def load_yaml(path):

    with open(path, 'r') as f:

        config = yaml.safe_load(f)

    return config


def build_parser(mode):

    parser = argparse.ArgumentParser()

    if mode == 'train':

        config = load_yaml(r'config/train/train_config.yaml')

        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', type=int, default=config['TRAIN']['SEED'])

        parser.add_argument('--device', type=str, default=config['TRAIN']['DEVICE'])

        parser.add_argument('--metric', type=str, default=config['TRAIN']['METRIC'])

        parser.add_argument('--load_checkpoint', type=bool, default=config['CHECKPOINTS']['LOAD_CHECKPOINT'])

        parser.add_argument('--checkpoint_path', type=str, default=config['CHECKPOINTS']['PATH'])

        # 添加 model 相关参数

        parser.add_argument('--model_id', type=str, default=config['MODEL']['MODEL_ID'])

        # 添加 trainer 相关参数

        parser.add_argument('--batch_size', type=int, default=config['TRAIN']['BATCH_SIZE'])

        parser.add_argument('--epochs', type=int, default=config['TRAIN']['MAX_EPOCH'])

        parser.add_argument('--early_stop_turns', type=int, default=config['TRAIN']['EARLY_STOP_TURNS'])

        # 添加 optim 相关参数

        parser.add_argument('--optim', type=str, default=config['OPTIM']['NAME'])

        parser.add_argument('--lr', type=float, default=config['OPTIM']['LR'])

        # 添加 数据集 相关参数

        parser.add_argument('--dataset_path', type=str, default=config['DATASET'][config['TRAIN']['DATASET']]['PATH'])

        parser.add_argument('--dataset_id', type=str, default=config['DATASET'][config['TRAIN']['DATASET']]['DATASET_ID'])

        parser.add_argument('--output_path', type=str, default=config['TRAIN']['OUTPUT_PATH'])

    elif mode == 'test':

        config = load_yaml(r'config/test/test_config.yaml')

        parser.add_argument('--seed', type=int, default=config['TEST']['SEED'])

        parser.add_argument('--device', type=str, default=config['TEST']['DEVICE'])

        parser.add_argument('--save', type=str, default=config['TEST']['SAVE_FOLDER'])

        parser.add_argument('--metric', type=list, default=config['TEST']['METRIC'])

        parser.add_argument('--batch_size', type=int, default=config['TEST']['BATCH_SIZE'])

        # 添加 model 相关参数

        parser.add_argument('--model_id', type=str, default=config['MODEL']['MODEL_ID'])

        # 添加 数据集 相关参数

        parser.add_argument('--dataset_path', type=str, default=config['DATASET'][config['TEST']['DATASET']]['PATH'])

        parser.add_argument('--dataset_id', type=str, default=config['DATASET'][config['TEST']['DATASET']]['DATASET_ID'])

        parser.add_argument('--model_path',type=str,default=config['MODEL']['TRAINED_MODEL_PATH'])

    return parser


if __name__ == '__main__':

    parser = build_parser('model')

    args = parser.parse_args()

    print(args)
