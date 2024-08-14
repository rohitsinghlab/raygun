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
    def __init__(self, reduce_size = 50, dim = 1280):
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
        gapleft = gapright = gap // 2
        if gap % 2 == 1:
            gapleft += 1
        mid = self.reduce_size - gap
        
        firstbeg, firstend = 0, gapleft * (min_window_size + 1)
        lastbeg, lastend   = seqs - gapright * (min_window_size + 1), seqs
        midbeg, midend     = gapleft * (min_window_size + 1), gapleft * (min_window_size + 1) + mid * min_window_size
        
        xstart = x[:, firstbeg:firstend, :]
        xmid  = x[:, midbeg:midend, :]
        xend  = x[:, lastbeg:lastend, :]
        
        xstmean = self.get_mean_std(xstart, min_window_size + 1, batch, dim,
                                               getstd, returnzero = (gapleft == 0), 
                                               device = x.device)
        xmidmean = self.get_mean_std(xmid, min_window_size, batch, dim,
                                                 getstd, returnzero = False,
                                                 device = x.device)
        xendmean = self.get_mean_std(xend, min_window_size + 1, batch, dim,
                                                 getstd, returnzero = (gapright == 0),
                                                 device = x.device)
        if getstd:
            xstmean, xststd   = xstmean
            xmidmean, xmidstd = xmidmean
            xendmean, xendstd = xendmean
            xmean = torch.concat([xstmean, xmidmean, xendmean], dim = 1)
            xstd = torch.concat([xststd, xmidstd, xendstd], dim = 1)
            return xmean, xstd
        else:
            xmean = torch.concat([xstmean, xmidmean, xendmean], dim = 1)
            return xmean
            

    def get_mean_std(self, x, windowsize, batch, dim,
                     getstd = False, returnzero = False, 
                     device = "cpu"):
        if returnzero:
            if getstd:
                return torch.zeros(batch, 0, dim).to(device), torch.zeros(batch, 0, dim).to(x.device)
            else:
                return torch.zeros(batch, 0, dim).to(device)
        xredmean = reduce(x, "b (x dx) c -> b x c", "mean",
                          dx = windowsize)
        if getstd:
            xdiff = x - repeat(xredmean, "b x c -> b (x dx) c", 
                          dx = windowsize)
            xdiffsq = xdiff * xdiff
            xmidstd = torch.sqrt(reduce(xdiffsq, "b (x dx) c -> b x c", "mean", 
                                       dx = windowsize))
            return xredmean, xmidstd
        else:
            return xredmean


"""
k * b - (k-1) * (b-p) = p
"""