# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
from torch.utils.data import Dataset, DataLoader
import esm
from glob import glob
import h5py 
from tqdm import tqdm
import re
from io import StringIO
import pandas as pd
import os
import torch
from Bio import SeqIO


class RaygunData(Dataset):
    def __init__(self, model, alphabet, 
                 fastafile, saveembedfolder = None, 
                 save = False,
                 device = "cpu", length = -1, config = None,
                 maxlength=3500,
                 prediction = False):
        self.fastafile = fastafile
        self.model = model
        self.alphabet = alphabet
        self.bc = self.alphabet.get_batch_converter()
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.eval()
            self.device = device
            self.computeembed = True
        else:
            self.computeembed = False
        self.pdbs = list(SeqIO.parse(fastafile, "fasta"))
        self.sequences = [(pdb.id, str(pdb.seq)) for pdb in self.pdbs if 
                         len(pdb.seq) <= maxlength]
            
        if length < 0:
            length = len(self.sequences)
        self.length = length
        self.saveembedfolder = saveembedfolder
        self.prediction = prediction
        self.save = save
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = [self.sequences[idx]]
        _, _, tokens = self.bc(data)
        
        # if embedfolder is present, check if the embeddings are already computed
        if self.saveembedfolder is not None:
            ifile = f"{self.saveembedfolder}/{self.sequences[idx][0]}.h5"
            if os.path.exists(ifile):
                with h5py.File(ifile, "r") as hf:
                    embedding = hf.get(self.sequences[idx][0])[:]
                    if self.prediction:
                        return self.sequences[idx][0], torch.from_numpy(embedding) #return name and embedding
                    embedding = torch.from_numpy(embedding)
                    return tokens.flatten(), embedding
        
        # If precompute is true and the mode is not prediction, raise this
        if self.computeembed is False and not self.prediction:
            raise Exception(f"embedding file {self.sequences[idx][0]}.h5 not present in {self.saveembedfolder}")
        # if precompute is false, then only construct the embeddings
        with torch.no_grad():
            tokens = tokens.to(self.device)
            embedding = self.model(tokens, repr_layers = [33],
                                   return_contacts = False)["representations"][33].to("cpu")
            if self.saveembedfolder is not None and self.save:
                with h5py.File(ifile, "w") as hf:
                    hf.create_dataset(self.sequences[idx][0], 
                                     data = embedding[0, 1:-1, :])
        embedding = embedding[0, 1:-1, :]
        if self.prediction:
            return self.sequences[idx][0], embedding # return name and embeddings
        return tokens.flatten(), embedding