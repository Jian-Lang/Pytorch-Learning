"""
@author: Lobster
@software: PyCharm
@file: utils.py.py
@time: 2024/3/27 20:31
"""
from angle_emb import AnglE
from transformers import  ViTModel


def load_vit_model(vit_path):

    model = ViTModel.from_pretrained(vit_path)

    return model


def load_angle_model(angle_path):

    angle_model = AnglE.from_pretrained(angle_path,
                                        pooling_strategy='cls_avg').cuda()

    return angle_model

