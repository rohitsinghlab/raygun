import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from esm.model.esm2 import TransformerLayer
from tqdm import tqdm

from raygun.decoder import DecoderBlock
from raygun.model_utils import Block
from raygun.reduction import Reduction
from raygun.repitition import Repitition

class RaygunEncoder(nn.Module):
    """
    Reduce variable-length sequence embedding into fixed-length representation.

    Encodes a fixed number of segments of the original sequence as their average
    values. Optionally incorporate error-scaling term to add standard deviation
    scaled variation into embeddings for regularization and variational representation.
    """
    def __init__(self, num_segments=50, embed_dim=1280, conv_kernel=7, num_heads=20,
                 num_encoders=2, dropout=0.2, activation='gelu'):
        super(RaygunEncoder, self).__init__()
        self.dropout = dropout

        # for multiple reduction steps
        self.encoders = nn.ModuleList()
