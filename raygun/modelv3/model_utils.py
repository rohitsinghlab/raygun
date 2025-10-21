# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import reduce, repeat, rearrange
import numpy as np
import os
import math
from esm.model.esm2 import TransformerLayer
from einops.layers.torch import Rearrange

class ConvMasked(nn.Module):
    """
    Applied modifications on Conv1d to make the masking work
    """
    def __init__(self, indim, outdim, kernel_size):
        super(ConvMasked, self).__init__()
        self.conv        = nn.Conv1d(indim, outdim, 
                                    kernel_size = kernel_size, padding = "valid")
        self.kernel_size = kernel_size
        self.indim       = indim

    def forward(self, x, mask = None):
        batch, _, _ = x.shape
        padding = torch.zeros(batch, self.indim, self.kernel_size - 1).to(x.device)
        x1      = torch.concat([x, padding], dim = 2)
        y       =  self.conv(x1)
        if mask is not None:
            y   = y * mask.unsqueeze(1) # unsqueeze the seequence part
        return y

class ConvBlock(nn.Module):
    def __init__(self, dim, convkernel):
        super(ConvBlock, self).__init__()
        self.c1 = ConvMasked(dim     , dim // 2, kernel_size=convkernel)
        self.s1 = nn.SiLU()
        self.c2 = ConvMasked(dim // 2, dim // 4, kernel_size=convkernel // 2)
        self.s2 = nn.SiLU()
        self.c3 = ConvMasked(dim // 4, dim // 2, kernel_size=convkernel)
        self.s3 = nn.SiLU()
        return
    
    def forward(self, x, mask = None): 
        x = rearrange(x, "b n c -> b c n")
        if mask is not None:
            x = x * mask.unsqueeze(1)
        x = self.s1(self.c1(x, mask))
        x = self.s2(self.c2(x, mask))
        x = self.s3(self.c3(x, mask))
        x = rearrange(x, "b c n -> b n c")
        return x

class Block(nn.Module):
    def __init__(self, dim = 2560, attnheads = 5, convkernel = 7):
        super(Block, self).__init__()
        self.encoder = TransformerLayer(embed_dim = dim, 
                                       ffn_embed_dim = 2 * dim,
                                       attention_heads = attnheads,
                                       use_rotary_embeddings = True)
        
        self.convblock = ConvBlock(dim, convkernel)
        self.final = nn.Linear(dim // 2, dim)
        
    def forward(self, x, mask = None):
        x    = rearrange(x, "b n c -> n b c")
        x, _ = self.encoder(x, self_attn_padding_mask = ~mask if mask is not None else mask) 
        x    = rearrange(x, "n b c -> b n c")

        x    = self.convblock(x, mask = mask)
        
        return self.final(x)
    
    
class BlockP(nn.Module):
    def __init__(self, dim = 2560, attnheads = 5, convkernel = 7):
        """
        Obsolete 
        """
        super(BlockP, self).__init__()
        self.encoder = TransformerLayer(embed_dim = dim, 
                                       ffn_embed_dim = 2 * dim,
                                       attention_heads = attnheads,
                                        use_rotary_embeddings = True
                                       )
        self.convblock = nn.Sequential(Rearrange("b n c -> b c n"),
                                       nn.Conv1d(dim, dim // 2, kernel_size = convkernel, padding = "same"),
                                       nn.SiLU(),
                                       nn.Conv1d(dim // 2, dim // 4, kernel_size = convkernel // 2, padding = "same"), 
                                       nn.SiLU(),
                                       nn.Conv1d(dim // 4, dim // 2, kernel_size = convkernel, padding = "same"),    
                                       Rearrange("b c n -> b n c"), 
                                       # nn.GELU(),
                                    )
        self.final = nn.Linear(dim // 2, dim)
        # self.norm1 = nn.LayerNorm(dim // 2)
        
    def forward(self, x, mask = None):
        """
        This is an older Block version which does not use mask. 
        """
        batch, seq, len = x.shape
        assert batch == 1, "Batch should be equal to one for this obsolete class"
        x = rearrange(x, "b n c -> n b c")
        x, _ = self.encoder(x)
        x = rearrange(x, "n b c -> b n c")
        x = self.convblock(x)
        # x = self.norm1(x)
        return self.final(x)