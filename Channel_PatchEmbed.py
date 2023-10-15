import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# David Channel Patch Embed

class ChannelPatchEmbed(nn.Module):
    """ A patch embedding procedure where embedding tokens are generated individually for patches of each channel in the
    input modalities. Employs a 2D depth-wise convolution to generate the individual tokens for each channel patch
    without significantly more trainable parameters than the default patch embedding procedure.

    :param img_size: The height and width of the input images.
    :param patch_size: The patch size of the patches.
    :param mod_1_channels: The number of input channels for the first modality.
    :param mod_2_channels: The number of input channels for the second modality,
                           should be zero if only one modality is provided.
    :param embed_dim: The embedding dimension of the patch embeddings to be generated.
    :param norm_layer: A normalisation procedure to be applied to the patch embeddings. If not provided the patch
    embeddings will not be normalised before passing them to a model.
    """
    def __init__(self, img_size=120, patch_size=20, mod_1_channels=2, mod_2_channels=10, embed_dim=256, norm_layer=None):
        """ Creates an instance.

        :param img_size: The height and width of the input images.
        :param patch_size: The patch size of the patches.
        :param mod_1_channels: The number of input channels for the first modality.
        :param mod_2_channels: The number of input channels for the second modality,
                               should be zero if only one modality is provided.
        :param embed_dim: The embedding dimension of the patch embeddings to be generated.
        :param norm_layer: A normalisation procedure to be applied to the patch embeddings. If not provided the patch
        embeddings will not be normalised before passing them to a model.
        """
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_chans = mod_1_channels + mod_2_channels
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size) * self.in_chans

        self.proj = nn.Conv2d(self.in_chans,
                              self.in_chans * self.embed_dim,
                              kernel_size=(patch_size, patch_size),
                              stride=(patch_size, patch_size),
                              groups=self.in_chans)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.proj(x)
        _, _, patches_x, patches_y = x.shape
        x = x.view(B, self.embed_dim, self.in_chans, patches_x, patches_y)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# Veronica Channel Patch Embed

class ChannelPatchEmbed(nn.Module):

    def __init__(self, patch_size=4, rgb_chans=3, hs_chans=4, dem_chans=1, embed_dim=96, norm_layer=None):

        super().__init__()

        patch_size = to_2tuple(patch_size) # (4,4)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = rgb_chans + hs_chans + dem_chans 

        self.proj = nn.Conv2d(1, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    

    def _padding(self, x, patch_size):

        _, _, H, W = x.size() # B, C, H, W
        if W % patch_size[1] != 0:
            x = F.pad(x, (0, patch_size[1] - W % patch_size[1]))
        if H % patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, patch_size[0] - H % patch_size[0]))
        
        return x   
    
    
    def forward(self, x):

        rgb, hs, dem = x
        x = torch.cat([rgb, hs, dem],1) # 16, 8, 256, 256

        outx = []
        
        for channel in range(0,self.in_chans): # controllare che passi su tutti i canali

            x_channel = x[:, channel, :, :] # select one channel --> B, 256, 256
            x_channel = torch.from_numpy(np.expand_dims(x_channel.cpu().detach().numpy(), axis=1)).cuda()
            x_channel = self._padding(x_channel, self.patch_size) # padding
            x_channel = self.proj(x_channel) # linear embedding --> B, 96, 64, 64
           
            if self.norm is not None:
                
                Wh, Ww = x_channel.size(2), x_channel.size(3) 
                x_channel = x_channel.flatten(2).transpose(1, 2) # B, 4096, 96
                x_channel = self.norm(x_channel) # B, 4096, 96
                x_channel = x_channel.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)  # B, 96, 64, 64
            
            outx.append(x_channel)
        
        x = torch.cat((outx[0], outx[1], outx[2], outx[3], outx[4], outx[5], outx[6], outx[7]), dim=1)
        
        return x