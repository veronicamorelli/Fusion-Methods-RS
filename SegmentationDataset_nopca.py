
from torch.utils.data import Dataset
import torch
import rasterio
import numpy as np
import pickle

# DATASET
class SegmentationDataset(Dataset):

    def __init__(self, 
                 path_rgb: str='path_rgb',
                 path_hs: str='path_hs',
                 path_dem: str='path_dem',
                 path_label: str='path_label', 
                 path_txt: str='path_txt', 
                 path_max_dict: str='path_max_dict',
                 path_min_dict: str='path_min_dict',
                 transform: bool=None): 

        self.path_rgb = path_rgb
        self.path_hs = path_hs
        self.path_dem = path_dem
        self.path_label = path_label
        self.txt = np.loadtxt(path_txt, dtype=str)

        # Normalizzazione
        self.path_min_dict = path_min_dict
        self.path_max_dict = path_max_dict

        self.transform = transform

    def import_toarray(self, path):
            im_raster = rasterio.open(path)
            return im_raster.read() # array
    
    def load_dict(self, path):
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict

    def __getitem__(self, index): 
    
        im = self.txt[index]

        path_im_rgb = self.path_rgb + im
        path_im_hs = self.path_hs + im
        path_im_dem = self.path_dem + im
        path_im_label = self.path_label + im

        im_array_rgb = self.import_toarray(path_im_rgb) 
        im_array_hs = self.import_toarray(path_im_hs) 
        im_array_dem = self.import_toarray(path_im_dem)
        label_array = self.import_toarray(path_im_label) 
                
        im_array_rgb = np.moveaxis(im_array_rgb, 0, -1) 
        im_array_hs = np.moveaxis(im_array_hs, 0, -1) 
        im_array_dem = np.moveaxis(im_array_dem, 0, -1)
        label_array = np.squeeze(label_array) 

        # Normalizzazione 1
        min_dict_before_norm = self.load_dict(self.path_min_dict)
        max_dict_before_norm = self.load_dict(self.path_max_dict)

        min_rgb = min_dict_before_norm['rgb']
        max_rgb = max_dict_before_norm['rgb']
        min_hs = min_dict_before_norm['hs']
        max_hs = max_dict_before_norm['hs']
        min_dem = min_dict_before_norm['dem']
        max_dem = max_dict_before_norm['dem']
        

        im_array_norm_rgb = ((im_array_rgb - min_rgb) / (max_rgb - min_rgb)) 
        im_array_norm_hs = ((im_array_hs - min_hs) / (max_hs - min_hs)) 
        im_array_norm_dem = ((im_array_dem - min_dem) / (max_dem - min_dem)) 

        if self.transform is not None:

            normalize_rgb, normalize_hs, normalize_dem, transforms_augmentation = self.transform
            rgb = normalize_rgb(image=im_array_norm_rgb)["image"] 
            hs = normalize_hs(image=im_array_norm_hs)["image"] 
            dem = normalize_dem(image=im_array_norm_dem)["image"] 

            transformed = transforms_augmentation(image=rgb, mask=label_array, hs=hs, dem=dem) 

            im_rgb = transformed["image"]
            im_hs = transformed["hs"]
            im_dem = transformed["dem"]
            mask = transformed["mask"] 

            samples = im_rgb, im_hs, im_dem

        mask[mask == 4] = 3
        mask[mask > 4] = mask[mask > 4] - 1

        return samples, mask.long(), im

    def __len__(self):
        return len(self.txt)