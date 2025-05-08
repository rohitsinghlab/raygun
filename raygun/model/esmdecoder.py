# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import torch
import torch.nn as nn
from esm.model.esm2 import TransformerLayer

class DecoderBlock(nn.Module):
    """
    Processes sequence embeddings through a Transformer layer and projects them
    to token vocabulary logits, typically for final token prediction.

    Used to go from input embedding representation back to token representation. Used
    when calculating cross-entropy loss and generating sequences.
    """
    def __init__(self, dim = 1280, nhead = 20, dropout = 0.2):
        super(DecoderBlock, self).__init__()
        # 'encoder' here refers to a standard TransformerLayer acting on decoder outputs
        self.encoder = TransformerLayer(embed_dim = dim, 
                                       ffn_embed_dim = 2 * dim,
                                       attention_heads = nhead,
                                       use_rotary_embeddings = True
                                       )
        # final layers to project to vocabulary size (32 in this case)
        self.final = nn.Sequential(nn.Linear(dim, dim // 4),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(dim // 4, 32))
    
    def load_pretrained(self, filename: str):
        """Loads pretrained weights for this DecoderBlock."""
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage) #ensure loading on CPU if GPU not avail.
        self.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint # Free memory
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate token logits."""
        # pass input through the transformer layer
        x, _ = self.encoder(x) 
        # project to final vocabulary logits
        return self.final(x)
