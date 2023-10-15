import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from Patch_embedding import DefaultPatchEmbed


# Utils (to understand)

# Copyright (c) OpenMMLab. All rights reserved.
def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


# DSA-concat

class DepthFusionModule1(MultiheadAttention):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super().__init__(
            embed_dims * 2,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)
        self.embed_dims = embed_dims
        self.gamma = Scale(0)
        self.norm = build_norm_layer(norm_cfg, embed_dims * 2)[1]

    def forward(self, color, depth):
        h, w = color.size(2), color.size(3)
        qkv = torch.cat([color, depth], dim=1)
        qkv = nchw_to_nlc(qkv)
        out = self.attn(query=qkv, key=qkv, value=qkv, need_weights=False)[0]
        out = self.gamma(out) + qkv
        color = nlc_to_nchw(self.norm(out), (h, w))[:, :self.embed_dims]
        return color





# DSA-add