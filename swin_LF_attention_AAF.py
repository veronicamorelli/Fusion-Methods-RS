# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from Patch_embedding import DefaultPatchEmbed_MF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Start of my code
######################################################################################################

# window_partition: nothing changed
# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

def window_partition(x, window_size):

    if len(x.shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    elif len(x.shape) == 3: 
        B, H, W = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 3, 2, 4).contiguous().view(-1, window_size, window_size)

    return windows

# window_reverse: nothing changed
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

##################################################################################################################

'''
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details. '''

import torch.nn as nn
import torch

# Token Substitution Process (Token Exchange)
class TokenExchange(nn.Module):

    def __init__(self):
        super(TokenExchange, self).__init__()

    def forward(self, im1, im2, mask1, mask2, mask_threshold):
        # x: [B, N, C], mask: [B, N, 1]

        x1, x2 = torch.zeros_like(im1), torch.zeros_like(im2)
        x1[mask1 >= mask_threshold] = im1[mask1 >= mask_threshold]
        x1[mask1 < mask_threshold] = im2[mask1 < mask_threshold]
        x2[mask2 >= mask_threshold] = im2[mask2 >= mask_threshold]
        x2[mask2 < mask_threshold] = im1[mask2 < mask_threshold]
        x = x1, x2
        return x

####################################################################################################################


# Window Attention: nothing changed
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        ''' Relative Position'''
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.attn_drop2 = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.exchange = TokenExchange()


    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        # x1: rgb
        # x2: hs

        x1, x2 = x
        x1_mask, x2_mask = mask
        B_, N, C = x1.shape

        qkv1 = self.qkv1(x1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv2(x2).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # make torchscript happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        q1 = q1 * self.scale
        attn1 = (q1 @ k1.transpose(-2, -1)) # 1600, 3, 49, 49

        q2 = q2 * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            (self.window_size[0] * self.window_size[1]), (self.window_size[0] * self.window_size[1]), -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.repeat(3,3,1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn1 = attn1 + relative_position_bias.unsqueeze(0) # 1600, 3, 147, 147
        attn2 = attn2 + relative_position_bias.unsqueeze(0)

        # if mask is not None:

        #     nW = mask.shape[0]
        #     attn1 = attn1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn1 = attn1.view(-1, self.num_heads, N, N)
        #     attn1 = self.softmax(attn1)

        #     attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn2 = attn2.view(-1, self.num_heads, N, N)
        #     attn2 = self.softmax(attn2)

        # else:

        #     attn1 = self.softmax(attn1)
        #     attn2 = self.softmax(attn1)

        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn1)

        attn1 = self.attn_drop1(attn1) # 1600, 3, 49, 49
        attn2 = self.attn_drop2(attn2)

        x1 = (attn1 @ v1).transpose(1, 2).reshape(B_, N, C) # 1600, 49, 96
        x1 = self.proj1(x1) # 1600, 49, 96
        x1 = self.proj_drop1(x1)

        x2 = (attn2 @ v2).transpose(1, 2).reshape(B_, N, C) # 1600, 147, 96
        x2 = self.proj2(x2) # 1600, 147, 96
        x2 = self.proj_drop2(x2)

        x = x1, x2

        if x1_mask is not None:

            x = x1, x2
            x1 = x1 * torch.unsqueeze(x1_mask, 2)
            x2 = x2 * torch.unsqueeze(x2_mask, 2)
            x = self.exchange(x1, x2, x1_mask, x2_mask, mask_threshold=0.2)

        return x

# Swin transformer block, everything change
class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # Da capire
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
    
    def forward(self, x, mask_matrix):

        rgb, hs = x
        # rgb: 8, 4096, 96
        # hs: 8, 4096, 96
        # mask_matrix: 8, 4096
        # mask: 8, 4096
        mask_matrix_rgb, mask_matrix_hs = mask_matrix 
        # mask_matrix = mask_matrix_rgb

        B, L, C = rgb.shape 

        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut_rgb = rgb # Shortcut: residual connection that will be added to the result of MLP(LN(x))
        shortcut_hs = hs

        rgb = self.norm1(rgb)
        hs = self.norm1(hs)

        rgb = rgb.view(B, H, W, C) # 8, 64, 64, 96
        hs = hs.view(B, H, W, C) # 8, 64, 64, 96
        mask_matrix_rgb = mask_matrix_rgb.view(B, H, W) # 8, 64, 64
        mask_matrix_hs = mask_matrix_hs.view(B, H, W) # 8, 64, 64

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        rgb = F.pad(rgb, (0, 0, pad_l, pad_r, pad_t, pad_b)) # 8, 70, 70, 96
        hs = F.pad(hs, (0, 0, pad_l, pad_r, pad_t, pad_b)) # 8, 70, 70, 96
        mask_rgb = F.pad(mask_matrix_rgb, (pad_l, pad_r, pad_t, pad_b)) # 8, 70, 70 
        mask_hs = F.pad(mask_matrix_hs, (pad_l, pad_r, pad_t, pad_b)) # 8, 70, 70
        
        _, Hp, Wp, _ = rgb.shape

        # cyclic shift
        if self.shift_size > 0:
        
            shifted_rgb = torch.roll(rgb, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_hs = torch.roll(hs, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) 
            shifted_mask_rgb = torch.roll(mask_rgb, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_mask_hs = torch.roll(mask_hs, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) 
            # attn_mask = mask_matrix 

            rgb_mask_windows = window_partition(shifted_mask_rgb, self.window_size)
            rgb_mask_windows = rgb_mask_windows.view(-1, self.window_size * self.window_size)
            
            hs_mask_windows = window_partition(shifted_mask_hs, self.window_size)
            hs_mask_windows = hs_mask_windows.view(-1, self.window_size * self.window_size)

        else: ############################## --> SISTEMARE MASK PER IL NONE
        
            shifted_rgb = rgb
            shifted_hs = hs

            rgb_mask_windows=None
            hs_mask_windows=None

        # partition windows
        rgb_windows = window_partition(shifted_rgb, self.window_size)  # nW*B, window_size, window_size, C
        rgb_windows = rgb_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        hs_windows = window_partition(shifted_hs, self.window_size)  # nW*B, window_size, window_size, C
        hs_windows = hs_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        #############################################

        # dim windows: 1600, 49, 96 (for each modality)
        # dim att_mask = 100, 49, 49 (for each modality)

        x_windows = rgb_windows, hs_windows
        mask_windows = rgb_mask_windows, hs_mask_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=mask_windows)  # nW*B, window_size*window_size, 

        attn_windows_rgb, attn_windows_hs = attn_windows # 800, 49, 96 each

        # merge windows
        attn_windows_rgb = attn_windows_rgb.view(-1, self.window_size, self.window_size, C) # 4800, 7, 7, 96
        attn_windows_hs = attn_windows_hs.view(-1, self.window_size, self.window_size, C) # 4800, 7, 7, 96

        shifted_rgb = window_reverse(attn_windows_rgb, self.window_size, Hp, Wp)  # B H' W' C --> 48, 70, 70, 96
        shifted_hs = window_reverse(attn_windows_hs, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            rgb = torch.roll(shifted_rgb, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            hs = torch.roll(shifted_hs, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            rgb = shifted_rgb
            hs = shifted_hs

        if pad_r > 0 or pad_b > 0:
            rgb = rgb[:, :H, :W, :].contiguous()
            hs = hs[:, :H, :W, :].contiguous()

        rgb = rgb.view(B, H * W, C)
        hs = hs.view(B, H * W, C)

        # FFN
        rgb = shortcut_rgb + self.drop_path(rgb)
        rgb = rgb + self.drop_path(self.mlp(self.norm2(rgb)))
        hs = shortcut_hs + self.drop_path(hs)
        hs = hs + self.drop_path(self.mlp(self.norm2(hs)))

        x = rgb, hs
        return x

        # return rgb, hs, dem


class PatchMerging(nn.Module): # il patch_merging si fa su un'immagine alla volta

    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    Patch merging layer concatenates features of each group of 2x2 neighboring patches
    LayerNorm is applied on the 4C-dimensional concatenated features
    (2x downsampling of resolution)
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) 
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B, L, C = x.shape # da capire da dove viene x
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x

######################################################################################

# Scoring Function
class PredictorLG(nn.Module):
    """ Image to Patch Embedding from DydamicVit
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1) # perché c'é logSoftmax e non Softmax?
        ) 

    def forward(self, x):
        x = self.score_nets(x)
        return x

#######################################################################################


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    
    Two Tansformer blocks together (Figure 3b in the paper)
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None, # andrá il PatchMerging
                 use_checkpoint=False): # to modify
        
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth # é una tupla
        self.use_checkpoint = use_checkpoint

        # self.embed_list = [dim, dim*2, dim*4, dim*8] # 96, 182, 384, 768 (ma non ne sono sicura perché forse dim cambia)
        # predictor_list = [PredictorLG(self.embed_list[i]) for i in range(len(self.embed_list))]
        # self.score_predictor = nn.ModuleList(predictor_list)

        self.embed_dim = dim
        self.score_predictor = PredictorLG(self.embed_dim)
        #embed_dim_list = [self.embed_dim, self.embed_dim*2, self.embed_dim*4, self.embed_dim*8 ]
        # self.score_predictor = nn.ModuleList([
        #     PredictorLG(embed_dim=self.embed_dim)
        # ])

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, # isinstance() check if drop_path is a list
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, i):
        """ Forward function.,

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        rgb, hs = x
        B = rgb.shape[0]

        # mask_list = []

        for blk in self.blocks: # blk blocks sono 4 blocchi di swin transformer (two successive swin transformer blocks)
            

            blk.H, blk.W = H, W

            ###########################################################################################################
            
            # MASK CREATION

            score_rgb = self.score_predictor(rgb).to(device) # 8, 4096, 2 | torch.Tensor | torch.float32
            score_hs = self.score_predictor(hs).to(device) # 8, 4096, 2 | torch.Tensor | torch.float32
            
            mask_rgb = F.softmax(score_rgb.reshape(B, -1, 2), dim=2)[:, :, 0] # 8, 4096
            mask_hs = F.softmax(score_hs.reshape(B, -1, 2), dim=2)[:, :, 0] # 8, 4096

            mask = mask_rgb, mask_hs
            # mask_for_list = [mask_rgb, mask_hs]
            # mask_list = mask_list.append(mask_for_list)

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, mask)
            else:
                x = blk(x, mask) 

            ############################################################################################################

        if self.downsample is not None:
            rgb, hs = x
            rgb_down = self.downsample(rgb, H, W)
            hs_down = self.downsample(hs, H, W)
            x_down = rgb_down, hs_down
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww #, mask_list
        else:
            return x, H, W, x, H, W #, mask_list
    
    # def attn_mask_creation(self, x, H, W):

    #     Hp = int(np.ceil(H / self.window_size)) * self.window_size # np.ceil return the ceiling of the input, element-wise (se ho 1.7 mi ritorna 1.)
    #     Wp = int(np.ceil(W / self.window_size)) * self.window_size
    #     img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
    #     h_slices = (slice(0, -self.window_size),
    #                 slice(-self.window_size, -self.shift_size),
    #                 slice(-self.shift_size, None))
    #     w_slices = (slice(0, -self.window_size),
    #                 slice(-self.window_size, -self.shift_size),
    #                 slice(-self.shift_size, None))
        
    #     cnt = 0
    #     for h in h_slices:
    #         for w in w_slices:
    #             img_mask[:, h, w, :] = cnt # che dim é img_mask? é della stessa dim di rgb e hs? posso crearla nel nuovo modo e fare poi window_partition?
    #             cnt += 1

    #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
    #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    #     return attn_mask


################################################################################################################
# End of my code


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=256, # 256
                 patch_size=4,
                 in_chans=[3, 4, 1],
                 embed_dim=96,
                 depths=[2, 2, 6, 2], # Swin-T
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.in_chans= in_chans
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        common_args_patch_embed = {'patch_size':patch_size, 
                                   'embed_dim':embed_dim, 
                                   'norm_layer':norm_layer if self.patch_norm else None}

        self.patch_embed_rgb = DefaultPatchEmbed_MF(in_chans=self.in_chans[0],
                                                    **common_args_patch_embed)
        self.patch_embed_hs = DefaultPatchEmbed_MF(in_chans=self.in_chans[1],
                                                    **common_args_patch_embed)

        # absolute positional embedding: linear embedding (dimension: C)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size) # (256, 256)
            patch_size = to_2tuple(patch_size) # (4, 4)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            # patches_resolution = [256//4, 254//4] = [64, 64]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            # (1, 96, 64, 64)
            
            """
            nn.Parameter: Tensor that is to be considered a module parameter
            Parameters are Tensor subclasses, that have a very special property when used with Module s - 
            when they are assigned as Module attributes they are automatically added to the list of its parameters, 
            and will appear e.g. in parameters() iterator. Assigning a Tensor does not have such effect. 
            This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model. 
            If there was no such class as Parameter, these temporaries would get registered too.

            numbers of tokens: 256/4x256/4 = 64x64 = 4096 
            each token: 4x4x27 --> feature dimension of each patch
            Patch Partition: H/4 x W/4 x 48 --> I have: H/4 x W/4 x 4x4x27 = 256/4 x 256/4 x 432
            ape (absolute positional embedding, linear embedding): applied to the raw vector ("token"/feature of a patch 4x4x27).
                It projectes the token to an embedding dimension (C). C=96 in this case. 
            To produce a hierarchical representation, the number is reduced by patch merging layers as the network gets deeper
            """

            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # num_layers=4 len(dephts)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, 
                use_checkpoint=use_checkpoint,
                )
            self.layers.append(layer)

            """
            Crea 4 layer (blocchi di swin transformers blocks)
            Ogni blocco contiene n blocchi di swin transformers block = n heads
                blocco 1: 3 blocchi swin transformer/heads
                blocco 2: 6 blocchi swin transformer/heads
                blocco 3: 12 blocchi swin transformer/heads
                blocco 4: 24 blocchi swin transformer/heads
            downsample: PatchMerging if i_layer < num_layers. Do PatchMerging 3 times 
            """

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')
        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""

        rgb, hs, _ = x
        rgb = self.patch_embed_rgb(rgb) 
        hs = self.patch_embed_hs(hs)

        Wh, Ww = rgb.size(2), rgb.size(3) # 64, 64

        if self.ape: # ape=False di default
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            rgb = (rgb + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            hs = (hs + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C

        else:
            rgb = rgb.flatten(2).transpose(1, 2)
            hs = hs.flatten(2).transpose(1, 2)

        rgb = self.pos_drop(rgb)
        hs = self.pos_drop(hs)

        x = rgb, hs

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]

            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, i) 
            #  x_out, H, W, x, Wh, Ww, mask_list = layer(x, Wh, Ww, i) 

            rgb_out, hs_out = x_out

            # rgb_outs = []
            # hs_outs = []

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}') # restituisce l'attributo "norm{i}"
                
                rgb_out = norm_layer(rgb_out)
                hs_out = norm_layer(hs_out)

                rgb_out2 = rgb_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                hs_out2 = hs_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                # rgb_outs.append(rgb_out2)
                # hs_outs.append(hs_outs2)

                out = torch.cat((rgb_out2, hs_out2), dim=1)

                outs.append(out)

                

        return outs # [rgb_outs, hs_outs], mask_list
        # outs is a list