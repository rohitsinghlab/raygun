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

class Reduction(nn.Module):
    def __init__(self, reduce_size = 50):
        self.reduce_size = reduce_size
        super(Reduction, self).__init__()
        return
    
    def forward(self, x, getstd = False):
        """
        return the sigma of noise from x, in addition to the reduction
        """
        batch, seqs, dim = x.shape
        min_window_size = seqs // self.reduce_size
        gap = seqs - min_window_size * self.reduce_size
        if gap % 2 == 0:
            starting_loc = (gap // 2)
        else:
            starting_loc = (gap // 2) + 1
            
        end_loc = starting_loc + min_window_size * self.reduce_size
        
        if starting_loc == 0:
            xstart = torch.zeros(batch, 0, dim).to(x.device)
            if getstd:
                xstartstd = torch.zeros(batch, 0, dim).to(x.device)
        else:
            xstart = x[:, :starting_loc]
            xstart = torch.mean(xstart, dim = 1, keepdims = True)
            if getstd:
                xstartstd = torch.std(xstart, dim = 1, correction = 0, 
                                      keepdims = True)
        
        xmid = x[:, starting_loc:end_loc]
        
        if end_loc == seqs:
            xend = torch.zeros(batch, 0, dim).to(x.device)
            if getstd:
                xendstd = torch.zeros(batch, 0, dim).to(x.device)
        else:
            xend = x[:, end_loc:]
            xend = torch.mean(xend, dim = 1, keepdims = True)
            if getstd:
                xendstd = torch.std(xend, dim = 1, correction = 0,
                                    keepdims = True)
            
        xmidreduced = reduce(xmid, "b (x dx) c -> b x c", "mean", dx = min_window_size)
        if getstd:
            xmiddiff   = xmid - repeat(xmidreduced, "b h c -> b (h repeats) c", repeats = min_window_size)
            xmiddiffsq = xmiddiff * xmiddiff
            xmidstd    = torch.sqrt(reduce(xmiddiffsq, "b (x dx) c -> b x c", "mean", dx = min_window_size))
        
        xcrr = torch.concat([xstart, xmidreduced[:, (0 if starting_loc == 0 else 1):
                                          (seqs if end_loc == seqs else -1), :], xend], dim = 1)
        if getstd:
            xcrrstd = torch.concat([xstartstd, xmidstd[:, (0 if starting_loc == 0 else 1): 
                                                      (seqs if end_loc == seqs else -1), :], xend], dim = 1)
            return (xmidreduced + xcrr) / 2, (xcrrstd + xmidstd) / 2
        
        return (xmidreduced + xcrr) / 2
    


class Repitition(nn.Module):
    def __init__(self):
        """
        self, noise_threshold
        """
        super(Repitition, self).__init__()
        return
        
    def forward(self, encoding, finallength):
        start = encoding[:, 0, :].unsqueeze(1)
        end = encoding[:, -1, :].unsqueeze(1)
        batch, encoderlength, dim = encoding.shape
        reps = finallength // encoderlength
        gap  = finallength % encoderlength
        """
        noise = torch.randn(<size>) * noise_threshold
        """
        
        if gap == 0:
            return repeat(encoding, f"b h c -> b (h {reps}) c")
        elif gap == 1:
            x1_last = repeat(encoding, f"b h c -> b (h {reps}) c")
            return torch.concat([start, x1_last], dim = 1)
        else:
            gapstart = gapend = gap // 2 
            if gap % 2 != 0:
                gapstart = gapstart + 1
            xstart = repeat(start, f"b h c -> b (h {gapstart}) c")
            xend = repeat(end, f"b h c -> b (h {gapend}) c")
            xmid = repeat(encoding, f"b h c -> b (h {reps}) c")
            return torch.concat([xstart, xmid, xend], dim = 1) # + noise

    
class Block(nn.Module):
    def __init__(self, dim = 2560, attnheads = 5, convkernel = 7):
        super(Block, self).__init__()
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
        
    def forward(self, x):
        x = rearrange(x, "b n c -> n b c")
        x, _ = self.encoder(x)
        x = rearrange(x, "n b c -> b n c")
        x = self.convblock(x)
        # x = self.norm1(x)
        return self.final(x)
