# Copyright 2024  Kapil Devkota, Rohit Singh
# Modified to use Change_length function Kavi Haria Shah 2025
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
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange
from raygun.modelv3.esmdecoder import DecoderBlock
from raygun.modelv3.model_utils import Block, BlockP
from raygun.modelv3.change import Change_length

class RaygunEncoder(nn.Module):
    def __init__(self, dim = 1280, reduction = 50,
                 convkernel = 7,
                 nhead = 20, numencoders = 2, dropout = 0.2,
                 activation = "gelu"):
        super(RaygunEncoder, self).__init__()
        self.dropout = dropout

        self.encoders = nn.ModuleList()
        for i in range(numencoders):
            self.encoders.append(Block(dim = dim, 
                                       convkernel = convkernel, 
                                       attnheads = nhead))


        self.redlength = reduction
        self.reduction = Change_length(finallength = reduction)
        self.final     = nn.Sequential(
                            nn.Linear(dim * (numencoders+1), dim // (numencoders + 1) * 4),
                            nn.SiLU(),
                            nn.Linear(dim // (numencoders + 1) * 4, dim)
                         )

        nn.init.xavier_uniform_(self.final[0].weight, gain=1e-3)
        nn.init.xavier_uniform_(self.final[2].weight, gain=1e-3)
        nn.init.constant_(self.final[0].bias, 0)
        nn.init.constant_(self.final[2].bias, 0)

    def reduce(self, x, mask = None, noise = None):
        """
        Use the Reduction operation to compress the PLM representation to a fixed-dimension space.
        """
        batch, _, _ = x.shape
        if noise is not None:
            redmean, redstd = self.reduction(x, mask = mask, getstd = True)
            reduced = redmean + torch.randn_like(redstd, device = x.device) * redstd * noise
        else:
            reduced = self.reduction(x, mask = mask, getstd = False)
        return reduced

    def forward(self, x, mask = None, noise = None, add_at_first_only=True):
        """
        If error_c is provided, noise component is incorporated into the 
        fixed-dimensional representation. 
        """
        enc = self.reduce(x, mask = mask, noise = noise)
        residues = [enc]
        for mod in self.encoders:
            xresidue = mod(x, mask = mask)
            residue  = mod(self.reduce(xresidue, mask = mask, 
                                       noise = None if add_at_first_only else noise)) # 
            x        = x + xresidue
            residues.append(residue)

        finalresidue = self.final(torch.concat(residues, dim = -1)) 
        return enc + finalresidue      


class RaygunDecoder(nn.Module):
    def __init__(self, dim = 1280, numdecoders = 5, convkernel = 7,
                 nhead = 20, dropout = 0.1, activation = "gelu"):
        super(RaygunDecoder, self).__init__()
        self.dbefore = nn.ModuleList()
         
        for i in range(numdecoders):
            self.dbefore.append(Block(dim = dim,
                                      convkernel = convkernel,
                                      attnheads = nhead,
                                      ))
        
        self.repetition = Change_length()

        self.dafter = nn.ModuleList()
        for i in range(numdecoders+1):
            self.dafter.append(Block(dim = dim,
                                      convkernel = convkernel, 
                                      attnheads = nhead, 
                                      ))
        self.final = nn.Sequential(
                            nn.Linear(dim * (numdecoders+2), dim // (numdecoders+2) * 4),
                            nn.SiLU(),
                            nn.Linear(dim // (numdecoders+2) * 4, dim)
                        )

    def forward(self, encoding, finallengths, mask = None):
        """
        Decoder is entirely deterministic. No noise added here.
        """
        out, _ = self.repetition(encoding, finallength=finallengths)
        # construct different encoding replicates
        ereplicates = []
        ereplicates.append(encoding)
        for mod in self.dbefore:
            encoding = encoding + mod(encoding, mask = mask)
            ereplicates.append(encoding)
        ## for each replicates, expand and apply model
        outreplicates = [out]
        for ereplicate, mod in zip(ereplicates, self.dafter):
            outreplicates.append(mod(self.repetition(ereplicate, finallengths),
                                    mask = mask))
        return out + self.final(torch.concat(outreplicates, dim = -1))

class Raygun(nn.Module):
    def __init__(self, dim = 1280, nhead = 20, convkernel = 7, 
                 numencoders = 10, numdecoders = 10,
                 dropout = 0.1,
                 reduction = 50, activation = "gelu",
                 esmdecodertotokenfile = None, 
                 fixed_esm_batching=False):
        super(Raygun, self).__init__()
        self.encoder = RaygunEncoder(dim     = dim, 
                                reduction    = reduction, 
                                convkernel   = convkernel,
                                numencoders  = numencoders, 
                                dropout      = dropout, 
                                activation   = activation,
                                nhead        = nhead)
        self.decoder = RaygunDecoder(dim     = dim, 
                                 nhead       = nhead, 
                                 convkernel  = convkernel,
                                 numdecoders = numdecoders,
                                 dropout     = dropout, 
                                 activation  = activation)

        self.esmdecoder = DecoderBlock(dim = dim, 
                                      nhead = 20, 
                                      fixed_batching=fixed_esm_batching)
        if esmdecodertotokenfile is not None:
            checkpoint = torch.load(esmdecodertotokenfile)
            self.esmdecoder.load_state_dict(checkpoint["model_state"])
            del checkpoint
        self.alphtotoks  = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
        self.esmalphdict = {i:k for k, i in self.alphtotoks.items()}

    def get_sequence_from_logits(self, logits, lengths,
                                temperature=None, include_valid_only=True):
        batch, seq, _ = logits.shape
        if batch == 1:
            assert isinstance(lengths, int) or lengths.shape[0] == 1, "batch=1 but multiple lengths provided"
            if isinstance(lengths, int):
                lengths = [lengths]
        else:
            assert len(lengths.shape) == 1 and lengths.shape[0] == batch, "batch size and `lengths` dimension should be the same"
        if include_valid_only:
            # include part of the tokens only corresponding to valid AAs
            toks_to_accept = list(range(4, 24))
        else:
            toks_to_accept = list(range(4, 29))
        alphdict       = {i: self.esmalphdict[k] for i, k in 
                             enumerate(toks_to_accept)}
        
        logits         = logits[:, :, toks_to_accept]
        
        output_seqs   = []
        with torch.no_grad():
            for idx, length in enumerate(lengths):
                logit   = logits[idx, :length, :]
                if temperature is None: 
                    # equivalent to temperature 0
                    ptokens = torch.argmax(logit, dim = -1).cpu().numpy().tolist()
                else:
                    assert isinstance(temperature, float) and temperature > 0, "temperature should be float and > 0"
                    ps      = torch.softmax(logit / temperature, dim=-1)
                    ptokens = torch.multinomial(ps, num_samples=1).squeeze().cpu().numpy().tolist()
                pseqs   = "".join([alphdict[t] for t in ptokens])
                output_seqs.append(pseqs)
        return output_seqs

    def get_sequences_from_fixed(self, fixedembs, lengths):
        with torch.no_grad():
            out    = self.decoder(fixedembs, lengths)
            logits = self.esmdecoder(out)
        return self.get_sequence_from_logits(logits, lengths)
    
    def forward(self, x, mask = None, 
                target_lengths = None, 
                noise = None, 
                token = None, 
                return_logits_and_seqs = False, 
                temperature=None, 
                include_valid_only=True):
        """
        Arguments:
        x    -> [batch, seq, dim]: ESM-2 650M embedding
        mask -> [batch, seq]: Binary matrix. Suppose the sequence length of a  `batch_id` is `n`. Then mask[batch_id] should be such that mask[batch_id, :n] = 1 and mask[batch_id, n:] = 0  
        output_lengths -> [batch]: target length
        """
        batch, length_, dim = x.shape
        if target_lengths is not None:
            assert target_lengths.shape[0] == batch, "`output_lengths` should be a 1d tensor, its dimension should match the batch size"
            lengths = target_lengths
        elif batch == 1:
            lengths = length_  
        else:
            assert mask is not None, "batch larger than 1 but mask is Null"
            lengths = mask.sum(dim = -1)
        mem = self.encoder(x, mask = mask, noise = noise)
        out = self.decoder(mem, lengths)
        
        result = {"fixed_length_embedding": mem, 
                 "reconstructed_embedding": out}
        
        if token is not None or return_logits_and_seqs:
            logits          = self.esmdecoder(out) #batch, seq, token
            result["logits"] = logits
            
        if token is not None:
            if len(token.shape) == 3:
                tok   = rearrange(token, "b h k -> (b h k)")
            else:
                tok   = rearrange(token, "b k -> (b k)")
            loss      = F.cross_entropy(rearrange(logits, "b h k -> (b h) k"), 
                                        tok, ignore_index = 1)
            result["ce_loss"] = loss
        if return_logits_and_seqs:
            result["generated-sequences"] = self.get_sequence_from_logits(logits, lengths, 
                                                                          temperature=temperature, 
                                                                          include_valid_only=include_valid_only)
        return result