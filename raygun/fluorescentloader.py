import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
import torch
import os
import esm
import numpy as np

class BrightnessData(Dataset):
    def __init__(self, fastafile, esmmodel, esmbc, 
                 device = "cpu", lengths = -1, mode = "training"):
        self.records = list(SeqIO.parse(fastafile, "fasta"))
        if lengths > 0:
            self.records = self.records[:lengths]
        self.length = len(self.records)
        self.esmmodel = esmmodel.to(device)
        self.device = device
        self.esmbc = esmbc
        self.mode = mode
        return
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        record = self.records[idx]
        name   = record.id
        seq    = str(record.seq)
        _, _, tokens = self.esmbc([(name, seq)])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            embedding = self.esmmodel(tokens, repr_layers = [33],
                                   return_contacts = False)["representations"][33]
        embedding = embedding[0, 1:-1, :]
        if self.mode == "training":
            brightness = record.description.split("=")[-1]
            return embedding, torch.tensor([float(brightness)], dtype = torch.float32).to(self.device)
        else: # prediction: will not be using dataloader for this
            return embedding.unsqueeze(0), record
        