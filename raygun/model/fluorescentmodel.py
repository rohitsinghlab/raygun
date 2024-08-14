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

class FluorescentClassifierHead(nn.Module):
    def __init__(self, dim = 1280, nhead = 10, dropout = 0.2):
        super(FluorescentClassifierHead, self).__init__()
        self.encoder = TransformerLayer(embed_dim = dim, 
                                        ffn_embed_dim = 2 * dim,
                                        attention_heads = nhead,
                                        use_rotary_embeddings = True
                                       )
        self.model = nn.Sequential(nn.Linear(dim, dim // 4),
                                  nn.Dropout(p=dropout),
                                  nn.ReLU(),
                                  nn.Linear(dim // 4, 32),
                                  nn.ReLU())
        self.final = nn.Sequential(
                        nn.Linear(32, 1),
                        nn.Sigmoid())
        return
    
    def load_pretrained(self, filename):
        checkpoint = torch.load(filename)["model_state_dict"]
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def forward(self, x, target = None):
        x = rearrange(x, "b n c -> n b c")
        x, _ = self.encoder(x)
        x = rearrange(x, "n b c -> b n c")
        output = self.model(x)
        final  = reduce(output, "b h k -> b k", reduction = "mean")
        final  = self.final(final)
        if target is not None:
            flattened = rearrange(final, "b h -> (b h)")
            targetflattened = rearrange(target, "b h -> (b h)")
            loss = F.binary_cross_entropy(flattened, targetflattened)
            return final, loss
        return final

class FluorescentHead(nn.Module):
    def __init__(self, dim = 1280, nhead = 10, dropout = 0.2):
        super(FluorescentHead, self).__init__()
        self.encoder = TransformerLayer(embed_dim = dim, 
                                       ffn_embed_dim = 2 * dim,
                                       attention_heads = nhead,
                                        use_rotary_embeddings = True
                                       )
        self.inter = nn.Sequential(nn.Linear(dim, dim // 4),
                                  nn.Dropout(p=dropout),
                                  nn.ReLU(),
                                  nn.Linear(dim // 4, 32),
                                  nn.ReLU())
        self.final = nn.Linear(32, 1)
        return
    
    def load_pretrained(self, filename):
        checkpoint = torch.load(filename)["model_state_dict"]
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def forward(self, x, target = None):
        x = rearrange(x, "b n c -> n b c")
        x, _ = self.encoder(x)
        x = rearrange(x, "n b c -> b n c")
        inter = self.inter(x)
        final = reduce(inter, "b h k -> b k", reduction = "mean")
        
        final = self.final(final).squeeze()
        if target is not None:
            return final, F.mse_loss(final, target.squeeze())
        return final


class SimpleFluorescentHead(nn.Module):
    def __init__(self, dim = 1280, dropout = 0.2):
        super(SimpleFluorescentHead, self).__init__()
        self.inter = nn.Sequential(nn.Linear(dim, dim // 4),
                                  nn.Dropout(p=dropout),
                                  nn.ReLU(),
                                  nn.Linear(dim // 4, 32),
                                  nn.ReLU())
        self.final = nn.Linear(32, 1)
        return
    
    def load_pretrained(self, filename):
        checkpoint = torch.load(filename)["model_state_dict"]
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def forward(self, x, target = None):
        inter = self.inter(x)
        final = reduce(inter, "b h k -> b k", reduction = "mean")
        
        final = self.final(F.relu(final)).squeeze()
        if target is not None:
            return final, F.mse_loss(final, target.squeeze())
        return final