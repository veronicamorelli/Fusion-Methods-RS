import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from itertools import chain

# from swin import SwinTransformer
# from swin_EF import SwinTransformer
from swin_MF_channel_level import SwinTransformer
from upernet import PSPModule, FPN_fuse

import json
with open('/home/veronica/Scrivania/RSIm/Fusion/Method_1/config.json', 'r') as f:
  config = json.load(f)

class UperNet_swin(nn.Module):

    # Implementing only the object path
    
    def __init__(self,
                 params:dict, 
                 num_classes=9): # c'erano altri parametri !
        
        super(UperNet_swin, self).__init__()

        self.config = params 

        common_args_backbone = {'pretrain_img_size':self.config["train_size"][0], 
                                'patch_size':self.config["patch_size"],
                                'embed_dim': self.config["embed_dim_MF_CL"],
                                'depths' : self.config["depth_S_B_L"],
                                'num_heads' : self.config["num_heads"],
                                'window_size' : self.config["window_size"],
                                'mlp_ratio' : self.config["mlp_ratio"],   
                                'drop_rate': self.config["drop_rate"],
                                'attn_drop_rate': self.config["attn_drop_rate"],
                                'drop_path_rate': self.config["drop_path_rate"],
                                'norm_layer':nn.LayerNorm,
                                'ape':False,
                                'patch_norm':True,
                                'qkv_bias':True,
                                'qk_scale':None,
                                'out_indices':(0, 1, 2, 3),
                                'use_checkpoint':False
                                # frozen_stages=-1
                                }

        self.backbone_rgb = SwinTransformer(in_chans=self.config["in_chans"][0],
                                            **common_args_backbone)
        
        self.backbone_hs = SwinTransformer(in_chans=self.config["in_chans"][1],
                                            **common_args_backbone)
        
        self.backbone_dem = SwinTransformer(in_chans=self.config["in_chans"][2],
                                            **common_args_backbone)

        feature_channels= self.config["feature_channels_upernet_MF_CL"]
        self.PPN = PSPModule(in_channels=feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)
    
    def _concat_features(self, features_rgb, features_hs, features_dem) -> list:
        list_features_out=[]
        for i in range(0,4):
            t_cat = torch.cat((features_rgb[i], features_hs[i], features_dem[i]), dim=1)
            list_features_out.append(t_cat)
        return list_features_out

    def forward(self, x):

        rgb = x[:, :3, :, :]
        hs = x[:, 3:7, :, :]
        dem = x[:, 7, :, :].unsqueeze(0)
        
        # rgb, hs, dem = x

        input_size = (self.config["train_size"][0], self.config["train_size"][1])

        features_rgb = self.backbone_rgb(rgb) # to check
        features_hs = self.backbone_hs(hs)
        features_dem = self.backbone_dem(dem)

        features_concat = self._concat_features(features_rgb, features_hs, features_dem) 
        # len(features) = 4
        # features[0].shape = torch.Size([16, 144, 64, 64])
        # features[1].shape = torch.Size([16, 188, 32, 32])
        # features[2].shape = torch.Size([16, 576, 16, 16])
        # features[3].shape = torch.Size([16, 1152, 8, 8])
        features_concat[-1] = self.PPN(features_concat[-1])
        x = self.head(self.FPN(features_concat)) # 16, 9, 64, 64

        x = F.interpolate(x, size=input_size, mode='bilinear') # 16, 9, 256, 256
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
