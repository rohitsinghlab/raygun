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

class Repetition(nn.Module):
    def __init__(self):
        """
        self, noise_threshold
        """
        super(Repetition, self).__init__()
        return
        
    def forward(self, encoding, finallengths):
        """
        encoding     => torch.Tensor [batch, REDUCTION_DIM, 1280]; Fixed dimensional representations
        finallengths => torch.Tensor [batch]; target lengths for each batch
        """
        batch, encoderlength, dim = encoding.shape

        if isinstance(finallengths, int):
            assert batch == 1
            finallengths = torch.tensor([finallengths], dtype = int)

        xs = []
        maxlength = torch.max(finallengths)

        for i in range(batch):
            finallength = finallengths[i]
            reps        = finallength // encoderlength
            gap         = finallength % encoderlength
            gapleft     = gapright = gap // 2

            if gap % 2 == 1:
                gapleft += 1
            
            mid = encoderlength - gap
            if gapleft == 0:
                xstart = torch.zeros(1, 0, dim).to(encoding.device)
            else:
                encstart = encoding[i, :gapleft, :].unsqueeze(0)
                xstart = repeat(encstart, f"b h c -> b (h rep) c", rep=reps+1)
            encmid = encoding[i, gapleft:gapleft + mid, :].unsqueeze(0)
            xmid = repeat(encmid, f"b h c -> b (h rep) c", rep = reps)
            if gapright == 0:
                xend = torch.zeros(1, 0, dim).to(encoding.device)
            else:
                encend = encoding[i, gapleft + mid:, :].unsqueeze(0)
                xend = repeat(encend, f"b h c -> b (h rep) c", rep = reps+1)
            padding  = torch.zeros(1, maxlength - finallength, dim).to(encoding.device)
            xs.append(torch.concat([xstart, xmid, xend, padding], dim = 1))
        return torch.concat(xs, dim = 0)