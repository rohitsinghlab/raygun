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
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from raygun.model.esmdecoder import DecoderBlock
from raygun.model.model_utils import Block
from raygun.model.reduction import Reduction
from raygun.model.repitition import Repitition

class RaygunEncoder(nn.Module):
    def __init__(self, dim = 1280, reduction = 50,
                 convkernel = 7,
                 nhead = 20, numencoders = 2, dropout = 0.2,
                 activation = "gelu"):
        super(RaygunEncoder, self).__init__()
        self.dropout = dropout
        
        block = Block
        
        self.encoders = nn.ModuleList()
        for i in range(numencoders):
            self.encoders.append(block(dim = dim, 
                                       convkernel = convkernel, 
                                       attnheads = nhead))
            
            
        self.redlength = reduction
        self.reduction = Reduction(reduce_size = reduction)
        self.final     = nn.Sequential(
                            nn.Linear(dim * (numencoders+1), dim // (numencoders + 1) * 4),
                            nn.SiLU(),
                            nn.Linear(dim // (numencoders + 1) * 4, dim)
                         )
        
        nn.init.xavier_uniform_(self.final[0].weight, gain=1e-3)
        nn.init.xavier_uniform_(self.final[2].weight, gain=1e-3)
        nn.init.constant_(self.final[0].bias, 0)
        nn.init.constant_(self.final[2].bias, 0)
    
    def reduce(self, x, error_c = None):
        if error_c is not None:
            redmean, redstd = self.reduction(x, getstd = True)
            reduced = redmean + torch.randn_like(redstd, device = x.device) * redstd * error_c
        else:
            reduced = self.reduction(x, getstd = False)
        return reduced
    
    def forward(self, x, error_c = None): 
        ### error only at the starting reduction operation. 
        ### need to further analyze the proper error_c value if 
        ### we added error on succeding reduction process as 
        ### well (Line 69)
        enc = self.reduce(x, error_c = error_c)
        residues = [enc]
        for mod in self.encoders:
            xresidue = mod(x)
            residue  = mod(self.reduction(xresidue)) 
            x        = x + xresidue
            residues.append(residue)
            
        finalresidue = self.final(torch.concat(residues, dim = -1)) 
        return enc + finalresidue      

class RaygunDecoder(nn.Module):
    def __init__(self, dim = 1280, numdecoders = 5, convkernel = 7,
                 nhead = 20, 
                 dropout = 0.1, activation = "gelu", use_esm_block = True):
        super(RaygunDecoder, self).__init__()
        block = Block    
        self.dbefore = nn.ModuleList()
        for i in range(numdecoders):
            self.dbefore.append(block(dim = dim,
                                      convkernel = convkernel,
                                      attnheads = nhead,
                                      ))
        
        self.repitition = Repitition()
        
        self.dafter = nn.ModuleList()
        for i in range(numdecoders+1):
            self.dafter.append(block(dim = dim,
                                      convkernel = convkernel, 
                                      attnheads = nhead, 
                                      ))
        self.final = nn.Sequential(
                            nn.Linear(dim * (numdecoders+2), dim // (numdecoders+2) * 4),
                            nn.SiLU(),
                            nn.Linear(dim // (numdecoders+2) * 4, dim)
                        )
    
    def forward(self, encoding, finallength):
        out = self.repitition(encoding, finallength)
        # construct different encoding replicates
        ereplicates = []
        ereplicates.append(encoding)
        for mod in self.dbefore:
            encoding = encoding + mod(encoding)
            ereplicates.append(encoding)
        ## for each replicates, expand and apply model
        outreplicates = [out]
        for ereplicate, mod in zip(ereplicates, self.dafter):
            outreplicates.append(mod(self.repitition(ereplicate, finallength)))
        return out + self.final(torch.concat(outreplicates, dim = -1))
    
class Raygun(nn.Module):
    def __init__(self, dim = 1280, nhead = 20, convkernel = 7, 
                 numencoders = 10, numdecoders = 10,
                 dropout = 0.1,
                 reduction = 50, activation = "gelu", use_esm_block = True,
                 esmdecodertotokenfile = None,
                 esm_alphabet = None):
        super(Raygun, self).__init__()
        self.encoder = RaygunEncoder(dim = dim, 
                                reduction = reduction, 
                                convkernel = convkernel,
                                numencoders = numencoders, 
                                dropout = dropout, 
                                activation = activation,
                                nhead = nhead)
        self.decoder = RaygunDecoder(dim = dim, 
                                 nhead = nhead, 
                                 convkernel = convkernel,
                                 numdecoders = numdecoders,
                                 dropout = dropout, 
                                 activation = activation)
        
        self.esmdecoder = DecoderBlock(dim = dim, 
                                      nhead = 20)
        if esmdecodertotokenfile is not None:
            checkpoint = torch.load(esmdecodertotokenfile)
            self.esmdecoder.load_state_dict(checkpoint["model_state"])
            del checkpoint
        self.esmalphdict = {i:k for k, i in esm_alphabet.items()}
        
    def load_pretrained(self, chkpoint):
        self.load_state_dict(chkpoint["model_state"])
        self.esmdecoder.load_state_dict(chkpoint["esmtotokensdecoder"])
        return
    
    def shrink(self, x, length, noise_c = None):
        self.eval()
        with torch.no_grad():
            mem = self.encoder(x)
            out = self.decoder(mem, length)
            logit  = rearrange(self.compute_loss(out), "b h k -> (b h) k")
            tokens = torch.argmax(logit, dim = -1).cpu().numpy()
            alph   = "".join([self.esmalphdict[x] for x in tokens])
        return alph
    
    def shrinkwithencoder(self, encoder, length):
        self.eval()
        with torch.no_grad():
            out = self.decoder(encoder, length)
            logit = rearrange(self.compute_loss(out), "b h k -> (b h) k")
            tokens = torch.argmax(logit, dim = -1).cpu().numpy()
            alph = "".join([self.esmalphdict[x] for x in tokens])
        return alph
    
    def compute_loss(self, out):
        return self.esmdecoder(out)
    
    def decode(self, mem, newlength):
        return self.decoder(mem, newlength)
    
    def get_blosum_score(self, embedding, true_token, config):
        with torch.no_grad():
            true_alph = config.convert_tokens_to_alph(true_token.flatten().cpu().numpy())
            logits = self.esmdecoder(embedding)
            pred_tokens = torch.argmax(logits, dim = -1).flatten().cpu().numpy()
            pred_alph   = config.convert_tokens_to_alph(pred_tokens)
        return config.compute_blosum_score(true_alph, pred_alph)

    def forward(self, x, token = None):
        batch, length, dim = x.shape
        mem = self.encoder(x)
        out = self.decoder(mem, length)
        if token is not None:
            logit = rearrange(self.compute_loss(out), "b h k -> (b h) k")
            if len(token.shape) == 3:
                tok   = rearrange(token, "b h k -> (b h k)")
            else:
                tok   = rearrange(token, "b k -> (b k)")
            loss  = F.cross_entropy(logit, tok)
            return out, mem, loss
        return out, mem
