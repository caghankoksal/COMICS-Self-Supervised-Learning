import os
import sys
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
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from utils import  *


class GoldenPanels_Jigsaw_Dataset(Dataset):
    def __init__(self,
                 images_path: str,
                 annotation_path : str,
                 permutation_path: str,
                 panel_dim,
                 shuffle: bool = True,
                 augment: bool = False,
                 train_test_ratio: float = 0.95,  # ratio of train data
                 train_mode: bool = True,
                 normalize = False,
                 num_panels = 1,
                 limit_size: int = -1,
                 num_tiles = 9,
                 num_classes = 200,
                 preload = False):

        self.images_path = images_path
        self.panel_dim = panel_dim
        self.augment = augment
        self.normalize = normalize
        self.num_panels = num_panels
        
        
        with open(annotation_path) as json_file:
            self.data = json.load(json_file)
        
        train_len = int(len(self.data) * train_test_ratio)

        if train_mode:
            self.data = self.data[:train_len]
        else:
            self.data = self.data[train_len:]

        if shuffle: # Be careful If you want panels to be selected sequentially, you should not shuffle.
            random.shuffle(self.data)
            
        if limit_size != -1:
            self.data = self.data[:limit_size]
            
        self.permutations   = np.load(permutation_path)
        self.permutations   = self.permutations-self.permutations.min()
        self.num_tiles = num_tiles
        self.preload = preload
        #self.adjust_format  = transforms.Compose([transforms.CenterCrop((216,168))])
        #self.adjust_format  = transforms.Compose([transforms.CenterCrop((216,168)),transforms.ToTensor(), transforms.Normalize(mean=means_CelebA, std=stds_CelebA)])

        self.tile_size      = (100,100)
        self.augment_tile   = transforms.Compose([transforms.CenterCrop(self.tile_size),
                                                  transforms.Lambda(color_jitter),
                                                  transforms.ToTensor()])
        
       
        self.avail_classes = [i for i in range(len(self.permutations))]
        
        
        
        if self.preload:
            self.all_images  = []
            for i in trange(len(self.data), desc='Preloading images to RAM...'):
                panel = Image.open(self.data[i]["path"])
                p_area = panel_sqrtize(*panel.size)
                panel = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
                self.all_images.append(panel)     
            
    
    def __getitem__(self,idx):
        if self.preload:
            img = self.all_images[idx]
        else:
            panel = Image.open(self.data[idx]["path"])
            p_area = panel_sqrtize(*panel.size)
            img = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
            img = TF.resize(panel, [360, 360])
        
    
        tile_size_x, tile_size_y = img.size[0]//3, img.size[1]//3
        tiles   = [None]*self.num_tiles
        
        #print("tile_size_x : ",tile_size_x,"tile_size_y : " ,tile_size_y, "size : ",img.size)
        
        
        
        for i in range(self.num_tiles):
            x,y  = i//3, i%3
            crop = [tile_size_x*x, tile_size_y*y]
            crop = [crop[0], crop[1], crop[0]+tile_size_x, crop[1]+tile_size_y]
            tile = img.crop(crop)
            tile = self.augment_tile(tile)
            tile_mean, tile_sd = tile.view(3,-1).mean(dim=1).numpy(), tile.view(3,-1).std(dim=1).numpy()
            tile_sd[tile_sd==0] = 1
            norm     = transforms.Normalize(mean=0.5, std=0.5)
            tile     = norm(tile)
            tiles[i] = tile
            
            
        rand_perm = np.random.randint(len(self.permutations))
        
        tiles = [tiles[self.permutations[rand_perm][i]] for i in range(self.num_tiles)]
        tiles = torch.stack(tiles,dim=0)

        return {'Tiles':tiles, 'Target':int(rand_perm), "tile_std": tile_sd, "tile_mean" : tile_mean, "path": self.data[idx]["path"]}
    
    
    def __len__(self):
        return len(self.data)
