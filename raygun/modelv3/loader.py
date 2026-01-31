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
import random
from collections import defaultdict
from io import StringIO
import pickle
import pandas as pd
import os
import torch
from einops import rearrange
from Bio import SeqIO
from torch.utils.data import BatchSampler

class RaygunData(Dataset):
    def __init__(self, datapath, alphabet=None, model = None,
                 precomputed = False, save = False,
                 embeddingfolder = None, 
                 device = "cpu", no_records = -1,
                 maxlength=1000, minlength=50,
                 batch_converter=None,
                 batch_preparer=None):
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
        # assert alphabet is not None, "ESM alphabet is not provided"
        ## NOTE: ESM-2 device location and `device` should be the same
        self.device          = device

        self.datapath = datapath
        self.model     = model
        self.alphabet  = alphabet

        self.bc = batch_converter

        with open(self.datapath, "rb") as f:
            self.records = pickle.load(f)

        self.sequences = []

        for group_id, df in self.records.items():
            for _, row in df.iterrows():
                seq = row["seq"]

                if not (minlength <= len(seq) <= maxlength):
                    continue

                seq_id = row.get("prot") or row.get("t-seq-id")

                self.sequences.append({
                    "id": seq_id,
                    "seq": seq,
                    "group_id": group_id,
                })

        self.group_to_indices = defaultdict(list)

        for idx, item in enumerate(self.sequences):
            gid = item["group_id"]
            self.group_to_indices[gid].append(idx)

        # Optional: drop tiny groups
        self.group_to_indices = {
            gid: idxs
            for gid, idxs in self.group_to_indices.items()
            if len(idxs) >= 4
        }

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
        self.batch_preparer  = batch_preparer
    
    def __len__(self):
        return self.no_records
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def collatefn_wo_esm(self, batches):
        ids, seqs  = zip(*batches)
        lengths    = [len(seq) for seq in seqs]
        maxlen     = max(lengths)
        nbatch     = len(lengths)
        mask       = torch.arange(maxlen, dtype = int).unsqueeze(0).expand(nbatch, maxlen) < torch.tensor(lengths, dtype = int).unsqueeze(1)
        tokens = []
        for b in batches:
            _, _, toks = self.bc([b]) # [1, seqlen]
            tokens.append(toks.squeeze(0))
        tokens       = pad_sequence(tokens, padding_value = 1)
        tokens       = rearrange(tokens, "s b -> b s")
        return tokens, mask, batches
    
    
    def collatefn(self, batches):
        embeddings            = []
        tokens, mask, batches = self.collatefn_wo_esm(batches)
        tokens                = tokens.to(self.device)
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

    def collatefn_with_e1(self, samples):
        """
        samples: list of raw sequences or dicts containing sequence strings.
        Returns dict with input_ids, attention_mask, lengths, raw.
        """
        seqs = []
        for s in samples:
            if isinstance(s, str):
                seqs.append(s)
            elif isinstance(s, dict):
                seqs.append(s.get("seq") or s.get("sequence") or s.get("raw_seq") or next(iter(s.values())))
            else:
                seqs.append(str(s))

        # encode using fast tokenizer backend
        encoded = self.batch_preparer.tokenizer.encode_batch(seqs)
        max_len = max(len(e.ids) for e in encoded)
        input_ids = torch.full((len(encoded), max_len), self.batch_preparer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, e in enumerate(encoded):
            ids = torch.tensor(e.ids, dtype=torch.long)
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1

        lengths = attention_mask.sum(dim=1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": lengths,
            "raw": samples
        }

class BatchSampler(BatchSampler):
    def __init__(
        self,
        group_to_indices: dict,
        batch_size: int = 4,
        batches_per_epoch: int | None = None,
        shuffle_groups: bool = True,
    ):
        self.group_to_indices = group_to_indices
        self.batch_size = batch_size
        self.groups = list(group_to_indices.keys())
        self.shuffle_groups = shuffle_groups

        # If not specified, define epoch length as "one pass over groups"
        self.batches_per_epoch = (
            batches_per_epoch if batches_per_epoch is not None else len(self.groups)
        )

    def __iter__(self):
        groups = self.groups.copy()
        if self.shuffle_groups:
            random.shuffle(groups)

        n = 0
        while n < self.batches_per_epoch:
            gid = random.choice(groups)
            indices = self.group_to_indices[gid]

            # Sample WITHOUT replacement (groups are large)
            batch = random.sample(indices, self.batch_size)

            yield batch
            n += 1

    def __len__(self):
        return self.batches_per_epoch
