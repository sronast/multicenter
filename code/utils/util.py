#
import os
import random
from pathlib import Path
#
import numpy as np
#
import torch
from torch.utils.data import DataLoader
#
from .datasets import SegDataset
#

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dir_lists(data_root):

    train_dir_polyp = data_root / 'train' /'image'
    val_dir_polyp =  data_root / 'val' / 'image'
    
    
    train_dir_list = list(train_dir_polyp.glob('*.tif'))
    val_dir_list = list(val_dir_polyp.glob('*.tif'))

    return train_dir_list, val_dir_list

def create_dataloader(data_path):
    # assert data in ['be', 'polyp', 'all'], print(f'No data named {data}')

    data_path = Path(data_path)
    train_dir_list, val_dir_list = get_dir_lists(data_path)
    
    train_dataset = SegDataset(train_dir_list, True)
    val_dataset = SegDataset(val_dir_list, False)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)
    return train_dataloader, val_dataloader