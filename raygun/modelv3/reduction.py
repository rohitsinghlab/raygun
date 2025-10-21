# Copyright 2024  Kapil Devkota, Rohit Singh
#Modified to replace reduction and repetition with Change_length function Copyright Kavi Haria Shah 2025
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
    
    def forward(self, x, mask = None, getstd = False):
        """
        return the sigma of noise from x, in addition to the reduction
        """
        batch, _, dim = x.shape

        if mask == None:
            ## batch, seqlen
            mask = torch.ones_like(x[:, :, 0], dtype = int)
        
        seqs     = torch.sum(mask, dim = 1) 
        xmeans   = []
        xstds    = []
        for i in range(batch):
            min_window_size  = seqs[i] // self.reduce_size
            gap              = seqs[i] - min_window_size * self.reduce_size
            gapleft          = gap // 2
            gapright         = gap // 2
            if gap % 2 == 1:
                gapleft += 1
            mid = self.reduce_size - gap
            
            firstbeg, firstend = 0, gapleft * (min_window_size + 1)
            lastbeg, lastend   = seqs[i] - gapright * (min_window_size + 1), seqs[i]
            midbeg, midend     = gapleft * (min_window_size + 1), gapleft * (min_window_size + 1) + mid * min_window_size
        
            xstart   = x[i, firstbeg:firstend, :].unsqueeze(0)
            xmid     = x[i, midbeg:midend, :].unsqueeze(0)
            xend     = x[i, lastbeg:lastend, :].unsqueeze(0)
            
            xstmean  = self.get_mean_std(xstart, min_window_size + 1, dim,
                                                getstd, returnzero = (gapleft == 0), 
                                                device = x.device)
            xmidmean = self.get_mean_std(xmid, min_window_size, dim,
                                                    getstd, returnzero = False,
                                                    device = x.device)
            xendmean = self.get_mean_std(xend, min_window_size + 1, dim,
                                                    getstd, returnzero = (gapright == 0),
                                                    device = x.device)
            if getstd:
                xstmean, xststd   = xstmean
                xmidmean, xmidstd = xmidmean
                xendmean, xendstd = xendmean
                xmean = torch.concat([xstmean, xmidmean, xendmean], dim = 1)
                xstd  = torch.concat([xststd, xmidstd, xendstd], dim = 1)
                xmeans.append(xmean)
                xstds.append(xstd)
            else:
                xmean = torch.concat([xstmean, xmidmean, xendmean], dim = 1)
                xmeans.append(xmean)
        if getstd:
            return torch.concat(xmeans, dim = 0), torch.concat(xstds, dim = 0)
        else:
            return torch.concat(xmeans, dim = 0)

    def get_mean_std(self, x, windowsize, dim,
                     getstd = False, returnzero = False, 
                     device = "cpu"):
        if returnzero:
            if getstd:
                return torch.zeros(1, 0, dim).to(device), torch.zeros(1, 0, dim).to(x.device)
            else:
                return torch.zeros(1, 0, dim).to(device)
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
