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
    """
    Reduce variable-length sequence embedding into fixed-length representation.

    Encodes a fixed number of segments of the original sequence as their average
    values. Optionally incorporate error-scaling term to add standard deviation
    scaled variation into embeddings for regularization and variational representation.
    """
    def __init__(self, reduce_size = 50, dim = 1280):
        self.reduce_size = reduce_size
        super(Reduction, self).__init__()
        return
    
    def forward(self, x, getstd = False):
        """
        Returns the reduced sequence representation and optionally the standard deviation
        used to scale variational noise (if getstd is True).
        """
        batch_size, sequence_len, embedding_dim = x.shape # Use more descriptive names for unpacked shape

        # calculate size of segments: base size and how many segments are larger for gap-filling
        min_window_size = sequence_len // self.reduce_size
        
        # for sequences not perfectly divisible by reduce_size, make segments
        # at beginning and end slightly larger (min_window_size + 1) to fill in the gap
        gap = sequence_len - min_window_size * self.reduce_size
        
        gapleft = gapright = gap // 2
        # if gap is not even, add one more larger segment on left
        if gap % 2 == 1:
            gapleft += 1
        
        # number of middle segments that will have the base 'min_window_size'
        mid = self.reduce_size - gap 
        
        # find start and end indices of the 3 groups (left, middle, right) in the whole sequence
        firstbeg, firstend = 0, gapleft * (min_window_size + 1)
        lastbeg, lastend   = sequence_len - gapright * (min_window_size + 1), sequence_len
        midbeg, midend     = gapleft * (min_window_size + 1), gapleft * (min_window_size + 1) + mid * min_window_size
        
        # get embeddings for 3 groups
        xstart = x[:, firstbeg:firstend, :]
        xmid  = x[:, midbeg:midend, :]
        xend  = x[:, lastbeg:lastend, :]
        
        # get mean and standard deviation representations of each segment group
        xstmean_processed = self.get_mean_std(xstart, min_window_size + 1, batch_size, embedding_dim,
                                              getstd, returnzero = (gapleft == 0), 
                                              device = x.device)
        xmidmean_processed = self.get_mean_std(xmid, min_window_size, batch_size, embedding_dim,
                                               getstd, returnzero = (mid == 0), # if mid count is 0
                                               device = x.device)
        xendmean_processed = self.get_mean_std(xend, min_window_size + 1, batch_size, embedding_dim,
                                               getstd, returnzero = (gapright == 0),
                                               device = x.device)
        
        # optionally return stds
        if getstd:
            xstmean, xststd   = xstmean_processed
            xmidmean, xmidstd = xmidmean_processed
            xendmean, xendstd = xendmean_processed
            
            xmean = torch.concat([xstmean, xmidmean, xendmean], dim = 1)
            xstd = torch.concat([xststd, xmidstd, xendstd], dim = 1)
            return xmean, xstd
        else:
            xstmean = xstmean_processed
            xmidmean = xmidmean_processed
            xendmean = xendmean_processed
            
            xmean = torch.concat([xstmean, xmidmean, xendmean], dim = 1)
            return xmean
            

    def get_mean_std(self, x_part: torch.Tensor, windowsize: int, batch_size: int, dim: int,
                     getstd: bool = False, returnzero: bool = False, 
                     device: torch.device = None): # Original 'device' was "cpu" default, now more general
        """
        Take a group of sequence embeddings (a part) and extract the mean and optionally 
        standard deviation of each segment within it as a compressed representation.
        """
        # if there are no segments in this part (e.g., gapleft was 0), then just return zeros
        if returnzero:
            # Original used 'device' (default "cpu" if not from x.device) for mean's zero tensor,
            # and x_part.device for std's zero tensor. This is preserved.
            # 'device' parameter here refers to the device of the original full input 'x'.
            mean_zeros_device = device if device is not None else torch.device("cpu") # Fallback for safety
            
            zeros_mean = torch.zeros(batch_size, 0, dim).to(mean_zeros_device)
            if getstd:
                # std zeros should be on the same device as x_part if it's not empty,
                # or fallback if x_part is unexpectedly problematic (though returnzero implies x_part is effectively empty for processing)
                std_zeros_device = x_part.device if x_part.numel() > 0 else mean_zeros_device
                zeros_std = torch.zeros(batch_size, 0, dim).to(std_zeros_device)
                return zeros_mean, zeros_std
            else:
                return zeros_mean
        
        # calculate mean per segment
        xredmean = reduce(x_part, "b (s l) c -> b s c", "mean", # s = number of segments in this part
                          l = windowsize) # l = windowsize (dx in original)
        
        # optionally return standard deviation
        if getstd:
            # this can represent the amount of information lost in the reduction
            xdiff = x_part - repeat(xredmean, "b s c -> b (s l) c", # s, l same as above
                                    l = windowsize)
            xdiffsq = xdiff * xdiff
            xmidstd = torch.sqrt(reduce(xdiffsq, "b (s l) c -> b s c", "mean", # s, l same as above
                                       l = windowsize))
            return xredmean, xmidstd
        else:
            return xredmean


"""
k * b - (k-1) * (b-p) = p
"""