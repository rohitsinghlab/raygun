# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import esm
from glob import glob
import h5py 
from tqdm import tqdm
import re
from io import StringIO
import pandas as pd
import os
import torch
from einops import rearrange
from Bio import SeqIO

class RaygunData(Dataset):
    def __init__(self, fastafile, alphabet, model = None,
                 precomputed = False, save = False,
                 embeddingfolder = None, 
                 device = "cpu", no_records = -1,
                 maxlength=1000, minlength=50):
        """
        parameters:
        model, alphabet => ESM-2 650M model and alphabet; ensure that it is in eval mode
        precomputed     => to indicate that the embeddings are precomputed
        save            => to save the computed embeddings
        embeddingfolder => if precomputed is True, it is the location where the embeddings are stored
                           if save is True, it is the location where the embeddings are saved
        no_records      => if positive, the number of items in the __getitem__ is overriden to the 
                           specified value
        maxlength       => maximum sequence length to allow
        """
        assert precomputed == False or embeddingfolder is not None, "precomputed is True but the `embeddingfolder` is not provided"
        assert save == False or embeddingfolder is not None, "save is True but the save location,  denoted by `embeddingfolder` is None"
        assert precomputed == True or model is not None, "precomputed is False, but the esm model is not provided"
        assert alphabet is not None, "ESM alphabet is not provided"
        ## NOTE: ESM-2 device location and `device` should be the same
        self.device          = device

        self.fastafile = fastafile
        self.model     = model
        self.alphabet  = alphabet
        self.bc        = self.alphabet.get_batch_converter()
        self.records   = list(SeqIO.parse(fastafile, "fasta"))
        self.sequences = [(rec.id, str(rec.seq)) for rec in self.records if 
                         len(rec.seq) <= maxlength and len(rec.seq) >= minlength]
        if precomputed:
            h5exists = lambda x : os.path.exists(f"{embeddingfolder}/{x}.h5")
            self.sequences = [s for s in self.sequences if h5exists(s[0])]
            self.save      = False            # no need to save if precomputed
        else:
            self.save      = save
        if no_records < 0:
            no_records = len(self.sequences)

        self.no_records      = no_records
        self.embeddingfolder = embeddingfolder
        self.precomputed     = precomputed
    
    def __len__(self):
        return self.no_records
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def collatefn(self, batches):
        ids, seqs  = zip(*batches)
        lengths    = [len(seq) for seq in seqs]
        maxlen     = max(lengths)
        nbatch     = len(lengths)
        mask       = torch.arange(maxlen, dtype = int).unsqueeze(0).expand(nbatch, maxlen) < torch.tensor(lengths, dtype = int).unsqueeze(1)
        embeddings = []
        # TOFIX: sometimes batch_converter adds padding token in the middle of the sequence
        tokens = []
        for b in batches:
            _, _, toks = self.bc([b]) # [1, seqlen]
            tokens.append(toks.squeeze(0))
        tokens       = pad_sequence(tokens, padding_value = 1)
        tokens       = rearrange(tokens, "s b -> b s")

        tokens       = tokens.to(self.device)
        if self.precomputed:
            for idx in ids:
                efile  = f"{self.embeddingfolder}/{idx}.h5"
                with h5py.File(efile, "r") as hf:
                    emb  = hf.get(idx)[:]
                    embeddings.append(torch.from_numpy(hf.get(idx)[:]).to(self.device))
            embeddings = pad_sequence(embeddings)
            embeddings = rearrange(embeddings, "n b c -> b n c")
        else:
            with torch.no_grad():
                embeddings = self.model(tokens, repr_layers = [33], 
                                        return_contacts = False)["representations"][33]
                embeddings = embeddings[:, 1:-1, :] # remove the start token
        if self.save:
            for i, idx, in enumerate(ids):
                efile = f"{self.embeddingfolder}/{idx}.h5"
                with h5py.File(efile, "w") as hf:
                    hf.create_dataset(idx, data = embeddings[0, :lengths[i], :].cpu().numpy())
        # remove start and end tokens
        tokens = tokens[:, 1:]
        tokens[tokens == 2] = 1 # 2 denotes eos

        return tokens[:, :-1].cpu(), embeddings.cpu(), mask, batches

