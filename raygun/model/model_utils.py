# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import torch.nn as nn
import torch
from einops import rearrange

from esm.model.esm2 import TransformerLayer
from einops.layers.torch import Rearrange

class Block(nn.Module):
    """
    Transformer and 1D-Convolution for mixing local and global sequence information.
    """
    def __init__(self, dim=2560, attnheads=5, convkernel=7):
        super(Block, self).__init__()
        
        # transformer layer
        self.encoder= TransformerLayer(embed_dim=dim,
                                            ffn_embed_dim=2*dim,
                                            attention_heads=attnheads,
                                            use_rotary_embeddings=True
                                            )
        # convolution
        self.convblock = nn.Sequential(Rearrange("b n c -> b c n"), # convolution expects batch_size, embed_dim, seq_len
                                  nn.Conv1d(dim, dim // 2, kernel_size=convkernel, padding="same"),
                                  nn.SiLU(),
                                  nn.Conv1d(dim // 2, dim // 4, kernel_size=convkernel // 2, padding="same"),
                                  nn.SiLU(),
                                  nn.Conv1d(dim // 4, dim // 2, kernel_size=convkernel, padding="same"),
                                  Rearrange("b c n -> b n c")
                                  )
        
        # final layer
        self.final = nn.Linear(dim // 2, dim)

    def forward(self, x):
        # apply transformer
        x = rearrange(x, "b n c -> n b c") # ESM transformer layers do not have "batch_first" option
        x, _ = self.encoder(x)
        x = rearrange(x, "n b c -> b n c")
        
        # apply convolution
        x = self.convblock(x)

        return self.final(x)
