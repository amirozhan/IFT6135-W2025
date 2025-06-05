from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os
import torchvision
from utils import closest_factors,sort_filters_and_find_inverses
import numpy as np


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        out = self.norm1(x)
        out =  out.permute(0,2,1)
        out = self.mlp_tokens(out)
        out = out.permute(0,2,1)

        out1 = x + out

        out = self.norm2(out1)
        out = self.mlp_channels(out)

        out = out + out1

        return out


class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """

        out = self.patchemb(images)

        for mixer_block in self.blocks:
            out = mixer_block(out)
        out = self.norm(out)
        out = out.mean(dim=1)

        out = self.head(out)

        # step1: Go through the patch embedding
        # step 2 Go through the mixer blocks
        # step 3 go through layer norm
        # step 4 Global averaging spatially
        # Classification

        return out
    
    
    def visualize(self, logdir):
        """ Visualize the token mixer layer 
        in the desired directory """

        w = self.blocks[0].mlp_tokens.fc1.weight.detach().cpu().permute(1,0)

        w, inverse_pairs, sorted_indices = sort_filters_and_find_inverses(w)


        new_order = list(inverse_pairs.values())  

        new_order_tensor = torch.tensor(new_order, dtype=torch.long)

        reordered_weights = w.index_select(0, new_order_tensor)
        xw, xy = closest_factors(reordered_weights.shape[1])  
        w_2d = reordered_weights.reshape(reordered_weights.shape[0], xw, xy)

        num_units = w_2d.shape[0]
        ncols = self.patchemb.grid_size  
        nrows = int(np.ceil(num_units / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.5, nrows*1.5))
        axes = axes.flatten()

        for i in range(num_units):
            ax = axes[i]
            ax.imshow(w_2d[i], cmap='coolwarm')
            old_idx = sorted_indices[i]
            ax.set_title(f"Weight index {old_idx}")
            ax.axis('off')

        for j in range(num_units, len(axes)):
            axes[j].axis('off')

    
        plt.tight_layout()
        plt.savefig(f"{logdir}/mlpmixer_weights.png", format='png')

        plt.show()

      
       
        



        

        
 
