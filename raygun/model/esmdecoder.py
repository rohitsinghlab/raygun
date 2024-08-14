# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import torch
import numpy as np
import os
import glob
from tqdm import tqdm 
import sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math
from esm.model.esm2 import TransformerLayer
from einops import repeat

class DecoderBlock(nn.Module):
    def __init__(self, dim = 1280, nhead = 20, dropout = 0.2):
        super(DecoderBlock, self).__init__()
        self.encoder = TransformerLayer(embed_dim = dim, 
                                       ffn_embed_dim = 2 * dim,
                                       attention_heads = nhead,
                                        use_rotary_embeddings = True
                                       )
        self.final = nn.Sequential(nn.Linear(dim, dim // 4),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(dim // 4, 32))
        return
    
    def load_pretrained(self, filename):
        checkpoint = torch.load(filename)["model_state_dict"]
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def forward(self, x):
        x, _ = self.encoder(x)
        return self.final(x)