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

# Model handler pytorch lighting
from model_handler import ModelHandler
# deterministic = "warn"

# Model
from upernet_swin_LateFusion import UperNet_swin

# torch.cuda.empty_cache()
seed_everything(42, workers=True)
AVAIL_GPUS = min(1, torch.cuda.device_count())

torch.use_deterministic_algorithms(True)
 
import json
with open('/home/veronica/Scrivania/RSIm/Fusion/Method_1/config.json', 'r') as f:
  config = json.load(f)

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

# DATA LOADER

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, worker_init_fn=seed_worker)
val_dataloader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=4, worker_init_fn=seed_worker)
test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=4, worker_init_fn=seed_worker)

'''
Situation with batch_size=2 in training. 
Size of training is not divisible by 2 prefectly (non restituisce un numero intero). 
The last batch is composed by just 1 image. F.interpolate returns error because it expects batch_size>1.
I add drop_last=True in training_dataloader to ignore the last incomplite batch. 
Usually, this problem does't raise because even if batch_size=n and total_number_of_sample/n is not integer, the training 
is done on a smaller subset of samples (n_samples_last_batch_size<n) but n_samples_last_batch_size would be always >1. 
'''

# MODEL

model = UperNet_swin(params = config,
                     num_classes=config["num_classes"])

# from focal_loss import FocalLoss
# weights = 1-torch.tensor([1-torch.finfo().eps, 0.08169914, 0.05278166, 0.16236057, 0.02600136, 0.31404807, 0.33979217, 0.02331703])
# criterion = FocalLoss(alpha=weights, ignore_index=0, gamma=1)

criterion = nn.CrossEntropyLoss(ignore_index=0)
# controllare se nel ModelHandler é necessario ripetere il nome
# optimizer = torch.optim.Adam(model_en_de.parameters(), lr=config["learning_rate"])

# lr_scheduler = ReduceLROnPlateau(optimizer, 
#                                  patience = config["patience"])

plt_model = ModelHandler(model= model,
                         criterion = criterion,
                         params = config, 
                         scheduler = True,
                         ) 

logger = TensorBoardLogger(save_dir=os.getcwd()) # da rivedere 
checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True) # monitor='val_loss'

# early_stop_callback = EarlyStopping(monitor="losses/val_loss", min_delta=0.00, patience=5, verbose=False, mode="max")

trainer = Trainer(
                  max_epochs=config["n_epochs"], 
                  accelerator = "auto", # da vedere se tenere
                  checkpoint_callback=checkpoint_callback,
                  #deterministic="warn",
                  logger=logger)

# TRAINING
trainer.fit(model=plt_model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader)

# TEST
trainer.test(dataloaders=test_dataloader) # ckpt_path = "best"

# Checkpoints

# checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
# print(checkpoint["hyper_parameters"])
# # {"learning_rate": the_value, "another_parameter": the_other_value}

# model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
# print(model.learning_rate)

# tensorboard --logdir . --bind_all

 