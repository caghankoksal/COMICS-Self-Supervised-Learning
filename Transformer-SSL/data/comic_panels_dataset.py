import os
import sys
import yaml
import json
from collections import namedtuple
from tqdm import tqdm
import copy
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .augment import panel_transforms, panel_squartize
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class PanelsDataset(Dataset):
    """
    Usage : 
    panels_dataset  = PanelsDataset(images_path = golden_age_config.panel_path,
                         annotation_path = golden_age_config.panels_annotation,
                         panel_dim = golden_age_config.panel_dim ,
                         num_panels = golden_age_config.num_panels,
                         train_test_ratio = golden_age_config.train_test_ratio,
                         normalize = False)
    dataloader = DataLoader(panels_dataset, batch_size=16, shuffle=False,
                num_workers=4)

    """

    def __init__(self,
                 images_path: str,
                 annotation_path: str,
                 panel_dim,
                 transformations=None,
                 shuffle: bool = True,
                 squartize: bool = True,
                 train_test_ratio: float = 0.95,  # ratio of train data
                 train_mode: bool = True,
                 limit_size: int = -1,
                 ):

        self.images_path = images_path
        self.annotation_path = annotation_path
        self.panel_dim = panel_dim
        self.transforms = transformations
        self.squartize = squartize

        with open(annotation_path) as json_file:
            self.data = json.load(json_file)

        train_len = int(len(self.data) * train_test_ratio)

        if train_mode:
            self.data = self.data[:train_len]
        else:
            self.data = self.data[train_len:]

        if shuffle:  # Be careful If you want panels to be selected sequentially, you should not shuffle.
            random.shuffle(self.data)

        if limit_size != -1:
            self.data = self.data[:limit_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        annot = self.data[idx]  # Dict with id serie panel information
        panel = Image.open(annot["path"]).convert('RGB')
        if self.squartize:
            panel = panel_squartize(panel, self.panel_dim)
        if self.transforms:
            panel = self.transforms(panel)
        '''
        if not self.squartize and not self.transforms:
            panel = transforms.ToTensor()(panel).unsqueeze(0)
        '''
        return panel
