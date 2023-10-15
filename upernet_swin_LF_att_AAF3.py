# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

# from swin import SwinTransformer
# from swin_EF import SwinTransformer
from swin_LF_attention_AAF3 import SwinTransformer
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

        self.backbone = SwinTransformer(pretrain_img_size= self.config["train_size"][0], 
                                        patch_size=self.config["patch_size"],
                                        in_chans = self.config["in_chans"],
                                        embed_dim = self.config["embed_dim_T_S"],
                                        depths = self.config["depths_T"],
                                        num_heads = self.config["num_heads"],
                                        window_size = self.config["window_size"],
                                        mlp_ratio = self.config["mlp_ratio"],
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop_rate= self.config["drop_rate"],
                                        attn_drop_rate= self.config["attn_drop_rate"],
                                        drop_path_rate= self.config["drop_path_rate"],
                                        norm_layer=nn.LayerNorm,
                                        ape=False,
                                        patch_norm=True,
                                        out_indices=(0, 1, 2, 3),
                                        # frozen_stages=-1,
                                        use_checkpoint=False)
        
        feature_channels= self.config["feature_channels_upernet_T_S"]
        self.PPN = PSPModule(in_channels=feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

        self.num_parallel = 2
        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))

    def forward(self, x):

        input_size = (self.config["train_size"][0], self.config["train_size"][1])

        # features = self.backbone(x)
        outputs_list, mask_list = self.backbone(x)
        
        output_list_fin = []
        for output in outputs_list:
            output[-1] = self.PPN(output[-1])
            x = self.head(self.FPN(output))
            x = F.interpolate(x, size=input_size, mode='bilinear')
            output_list_fin.append(x)

        # output_list_fin
        
        ens = 0
        alpha_soft = F.softmax(self.alpha)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * output_list_fin[l] # .detach()
        output_list_fin.append(ens)

        return output_list_fin, mask_list

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
