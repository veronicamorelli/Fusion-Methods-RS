import numpy as np
import random 
import torch
import os
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset
from SegmentationDataset import SegmentationDataset

# Utils
from utils import load_dict

from upernet_swin_CF import UperNet_swin
import json

with open('/home/veronica/Scrivania/RSIm/Fusion/Method_1/config.json', 'r') as f:
  config = json.load(f)

model = UperNet_swin(params = config,
                     num_classes=config["num_classes"])
# new_model = model.load_from_checkpoint(checkpoint_path="/home/veronica/Scrivania/RSIm/Fusion/Method_1/lightning_logs/ChannelFusionPatchEmbed1/checkpoints/epoch=249-step=33000.ckpt")

# LOAD and DATA AUGMENTATION
path_train_rgb = config["path_train_rgb"]
path_val_rgb = config["path_val_rgb"]
path_test_rgb = config["path_test_rgb"]

path_train_hs = config["path_train_hs"] 
path_val_hs = config["path_val_hs"]
path_test_hs = config["path_test_hs"]

path_train_dem = config["path_train_dem"]
path_val_dem = config["path_val_dem"]
path_test_dem = config["path_test_dem"]

path_label = config["path_label"]
path_train_txt = config["path_train_txt"]
path_val_txt = config["path_val_txt"]
path_test_txt = config["path_test_txt"]

path_max_dict = config["path_max_dict"]
path_min_dict = config["path_min_dict"]
path_max_dict01 = config["path_max_dict01"]
path_mean_dict01 = config["path_mean_dict01"]
path_std_dict01 = config["path_std_dict01"]

# LOAD DICTIONARIES
max_dict_01 = load_dict(path_max_dict01)
mean_dict_01 = load_dict(path_mean_dict01)
std_dict_01 = load_dict(path_std_dict01)

# DATA AUGMENTATION
# Data augmentation Albumentations
normalize_rgb = A.Normalize(mean = mean_dict_01['rgb'], std = std_dict_01['rgb'], max_pixel_value = max_dict_01['rgb'])
normalize_hs = A.Normalize(mean = mean_dict_01['hs'], std = std_dict_01['hs'], max_pixel_value = max_dict_01['hs'])
normalize_dem = A.Normalize(mean = mean_dict_01['dem'], std = std_dict_01['dem'], max_pixel_value = max_dict_01['dem'])


transform_composed_alb_train = A.Compose([A.Resize(height=config["input_size"][0], width=config["input_size"][1]),
                                          A.augmentations.crops.transforms.RandomCrop(config["train_size"][0], config["train_size"][1]),
                                          A.Rotate(limit = 90),
                                          A.HorizontalFlip(p = config["p_aug"]),
                                          A.VerticalFlip(p = config["p_aug"]),
                                          A.Transpose(p = config["p_aug"]),
                                          ToTensorV2()],
                                          additional_targets={
                                            "hs": "image",
	                                          "dem": "image"
					                                })

transform_composed_alb_val = A.Compose([A.Resize(height=config["train_size"][0], width=config["train_size"][1]), # é train_size giusto?
                                        ToTensorV2()],
                                          additional_targets={
                                            "hs": "image",
	                                          "dem": "image"
					                                })

transforms_train = normalize_rgb, normalize_hs, normalize_dem, transform_composed_alb_train
transforms_val = normalize_rgb, normalize_hs, normalize_dem, transform_composed_alb_val

# DATASET
train_set = SegmentationDataset(path_train_rgb, path_train_hs, path_train_dem, path_label, path_train_txt, path_max_dict, path_min_dict, transform=transforms_train) 
val_set = SegmentationDataset(path_val_rgb, path_val_hs, path_val_dem, path_label, path_val_txt, path_max_dict, path_min_dict, transform=transforms_val) 
test_set = SegmentationDataset(path_test_rgb, path_test_hs, path_test_dem, path_label, path_test_txt, path_max_dict, path_min_dict, transform=transforms_val) 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, worker_init_fn=seed_worker)
val_dataloader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=4, worker_init_fn=seed_worker)
test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=4, worker_init_fn=seed_worker)

model = UperNet_swin(params = config,
                     num_classes=config["num_classes"])
  # Create an instance of your model.
checkpoint_path = '/home/veronica/Scrivania/RSIm/Fusion/Method_1/lightning_logs/ChannelFusionPatchEmbed1/checkpoints/epoch=249-step=33000.ckpt'
checkpoint = torch.load(checkpoint_path)
model_state_dict = model.state_dict()
pretrained_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
model.load_state_dict(pretrained_state_dict)
