import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from scipy import stats

from torch import nn
from typing import Union, List, Dict
from operator import itemgetter
from datetime import datetime


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def read_json(file_path: str, encoding: str = 'utf-8', mode: str = 'r') -> Union[List[Dict], Dict]:
    """
    A helper function for reading json files.
    Args:
        file_path: The path to the file.
        encoding: The encoding format.
        mode: The open mode.

    Returns:
        A list of dictionaries or a dictionary.
    """
    with open(file_path, mode, encoding=encoding) as fr:
        return json.load(fr)


def save_json(json_obj: Union[List[Dict], Dict], file_path: str, mode: str = 'w', encoding: str = 'utf-8') -> None:
    """
    A function that writes the json object to a specific json file with given mode and encoding.
    Args:
        json_obj: A list of dict or a dict.
        file_path: The file path.
        mode: The mode while opening a file.
        encoding: The encoding format.

    Returns:
        Nothing.
    """
    json.dump(json_obj, open(file_path, mode, encoding=encoding), ensure_ascii=False, indent=4)


def find_head_indices(source, target, max_len=None):
    indices = []
    target_len = len(target)
    max_len = max_len if max_len else len(source)

    for i in range(max_len):
        if source[i: i + target_len] == target:
            indices.append(i)
    return indices


def mask_focal_loss(pred, gold, mask, gamma=2.0, alpha=0.25):
    ce_loss = F.binary_cross_entropy(pred, gold, reduction='none')
    p_t = pred * gold + (1 - pred) * (1 - gold)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * gold + (1 - alpha) * (1 - gold)
        loss = alpha_t * loss

    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    return torch.sum(loss * mask) / torch.sum(mask)


def mask_binary_cross_entropy_loss(pred, gold, mask):
    loss = F.binary_cross_entropy(pred, gold, reduction='none')
    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    return torch.sum(loss * mask) / torch.sum(mask)


def mask_cross_entropy_loss(pred, gold, mask):
    loss = F.cross_entropy(pred, gold, reduction='none')
    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    return torch.sum(loss * mask) / torch.sum(mask)


def get_gpu_memory_occupation(current_device):

    memory_occupation = 0
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation and try running on a GPU.")
        return memory_occupation

    allocated_memory = torch.cuda.memory_allocated(current_device)
    memory_occupation = allocated_memory / (1024 ** 2)

    return memory_occupation

def get_best_score(metrics_file, key:str="f1 score"):
    results = read_json(metrics_file)
    sorted_data = sorted(results, key=itemgetter(key), reverse=True)

    return sorted_data[0]
