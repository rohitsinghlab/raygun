# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import torch
import numpy as np
import os
import glob
from tqdm import tqdm 
import sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from esm.model.esm2 import TransformerLayer
from E1.modeling import DecoderLayer
from E1.config import E1Config


class E1DecoderBlock(nn.Module):
    """
    Drop-in replacement for esmdecoder.DecoderBlock

    Input:
        hidden_states: (B, S, 1280)
        within_seq_position_ids: (B, S)
        global_position_ids: (B, S)
        sequence_ids: (B, S)

    Output:
        logits: (B, S, 34)
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 34,
        dropout: float = 0.2,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.config = E1Config()

        self.input_proj = nn.Linear(input_dim, self.config.hidden_size)

        self.decoder = DecoderLayer(self.config, layer_idx)

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
            nn.Dropout(p=dropout),
            nn.Linear(self.config.hidden_size // 4, num_classes),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:

        B, S, _ = hidden_states.shape
        device = hidden_states.device

        if within_seq_position_ids is None:
            within_seq_position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        if global_position_ids is None:
            global_position_ids = within_seq_position_ids

        if sequence_ids is None:
            sequence_ids = torch.zeros((B, S), device=device, dtype=torch.long)

        hidden_states = self.input_proj(hidden_states)

        hidden_states, _, _ = self.decoder(
            hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

        return self.head(hidden_states)
