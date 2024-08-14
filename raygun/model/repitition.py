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

class Repitition(nn.Module):
    def __init__(self):
        """
        self, noise_threshold
        """
        super(Repitition, self).__init__()
        return
        
    def forward(self, encoding, finallength):
        batch, encoderlength, dim = encoding.shape
        reps = finallength // encoderlength
        gap  = finallength % encoderlength
        gapleft = gapright = gap // 2
        if gap % 2 == 1:
            gapleft += 1
        mid = encoderlength - gap
        if gapleft == 0:
            xstart = torch.zeros(batch, 0, dim).to(encoding.device)
        else:
            encstart = encoding[:, :gapleft, :]
            xstart = repeat(encstart, f"b h c -> b (h rep) c", rep=reps+1)
        encmid = encoding[:, gapleft:gapleft + mid, :]
        xmid = repeat(encmid, f"b h c -> b (h rep) c", rep = reps)
        if gapright == 0:
            xend = torch.zeros(batch, 0, dim).to(encoding.device)
        else:
            encend = encoding[:, gapleft + mid:, :]
            xend = repeat(encend, f"b h c -> b (h rep) c", rep = reps+1)
        return torch.concat([xstart, xmid, xend], dim = 1)
        