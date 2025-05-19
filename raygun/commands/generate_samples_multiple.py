# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
import sys
from raygun.modelv2.raygun import Raygun
from raygun.modelv2.esmdecoder import DecoderBlock
from raygun.modelv2.loader import RaygunData
from raygun.modelv2.ltraygun import RaygunLightning
from raygun.modelv2.pretrained import load_pretrained_may_16
from raygun.pll import get_PLL, penalizerepeats
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import esm
import os
import pandas as pd
import itertools
import json
import random
import re
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import subprocess
from collections import defaultdict
from Bio.Align import substitution_matrices
import shlex
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_cycles(embedding, finallength, model, nratio,
               numcycles = 0, esmmodel = None, esmbc = None, 
               device = 0):
    batch, seq, dim = embedding.shape
    assert batch == 1, "Batch size should be 1 for generation"
    encoder    = model.encoder(embedding, 
                               error_c = nratio)
    changedseq = model.get_sequences_from_fixed(encoder, finallength)[0]
    if numcycles == 0: return changedseq
    assert esmmodel is not None and esmbc is not None
    
    with torch.no_grad():
        esmmodel.eval()
        model.eval()
        for cycle in range(numcycles):
            data         = [("prot", changedseq)]
            _, _, tokens = esmbc(data)
            tokens       = tokens.to(device)
            embedcycle   = esmmodel(tokens, repr_layers = [33],
                                 return_contacts = False)["representations"][33][:, 1:-1, :]
            embedcycle = model.encoder(embedcycle) # do not add noise here
            changedseq = model.get_sequences_from_fixed(embedcycle, finallength)[0]
    return changedseq


@hydra.main(config_path="configs/", config_name="config", 
           version_base=None)
def main(config: DictConfig):
    config = OmegaConf.to_container(config, resolve = True)
    logger.info("Started the Raygun generation process")
    logger.info(f"Penalizerepeats set to {config['penalizerepeats']}.")
    logger.info(f"Length-agnostic PLL filtering activated. Filter ratio: {config['sample_ratio']}")
    logger.info(f"Sample fasta file: {config['templatefasta']}")
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esmmodel              = esmmodel.to(config["device"])
    bc                    = esmalphabet.get_batch_converter()
    
    # load the model 
    raymodel              = load_pretrained_may_16(local=True, 
                                                   localurl="/hpc/home/kd312/projects/raygunv2/src/raygun-new-publication/raygun")
    raymodel              = raymodel.to(config["device"])
    Path(config["sample_out_folder"]).mkdir(exist_ok=True)
    
    logger.info("Start Raygun sampling:")
    preddata = RaygunData(fastafile = config["templatefasta"],
                          alphabet  = esmalphabet,
                          model     = esmmodel,
                          device    = config["device"])
    print(f"\t\tNo of sequences to generate: {len(preddata)}")
    predloader = DataLoader(preddata, batch_size = 1, shuffle = False,
                           collate_fn = preddata.collatefn)
    
    shrinkeddata = []
    averaged_encoder = 0
    
    noiseratio = config["noiseratio"]
    pllaccept  = config["num_raygun_samples_to_generate"]
    togenerate = pllaccept * config["sample_ratio"]
    
    with open(config["lengthinfo"], "r") as js:
        lengthinfo = json.load(js) 
    
    records = []
    outprefix = f"{config['sample_out_folder']}/unfiltered_{noiseratio}_{togenerate}"

    logging.info("Raygun sampling started.")
    nameassignment = {}
    with torch.no_grad():
        raymodel.eval()
        for tok, emb, mask, batches in predloader:
            emb  = emb.to(config["device"])
            name = batches[0][0]
            for h in tqdm(range(togenerate)):
                nratio      = (noiseratio if (not config["randomize_noise"]) else 
                               random.random() * noiseratio)
                length      = np.random.randint(lengthinfo[name][0], lengthinfo[name][1]+1)
                changedseq = get_cycles(emb, length, raymodel,  
                                        nratio,
                                        numcycles = config["numcycles"],
                                        esmmodel = esmmodel, esmbc = bc)
                genname = f"{name}_i_{h}_l_{length}_n_{nratio}"
                nameassignment[genname] = name
                record = SeqRecord(Seq(changedseq),
                                   id = genname,
                                   description = f"noise ratio added: {nratio}")
                records.append(record)
    SeqIO.write(records, f"{outprefix}.fasta", "fasta")

    del raymodel
    del preddata
    
    # filter by pll
    plldf    = []
    plls     = defaultdict(list)
    for record in tqdm(records, desc = "Computing pll"):
        name = record.id
        seq  = str(record.seq)
        len_ = len(seq)
        pll  = get_PLL(seq, esmmodel, esmalphabet, bc)
        
        # adjusted pll
        pll  = pll / abs(-0.406 * len_ + 1.363)        

        # penalized repeats
        if config["penalizerepeats"]:
            pll = pll * penalizerepeats(seq)
        
        plldf.append((name, len_, pll, seq))
        plls[nameassignment[name]] += [(name, pll)]
    plldf = pd.DataFrame(plldf, columns = ["name", "length", "pll", "sequence"])
    plldf.to_csv(f"{outprefix}.pll.tsv", sep = "\t")

    filteredplls = []
    for key, val in plls.items():
        filteredplls += [x[0] for x in sorted(val, 
                                key = lambda x : x[1], 
                                reverse = True)[:pllaccept]] # find within buckets, samples with highest plls 
    filteredrecords = [record for record in records if record.id
                      in filteredplls]
    
    outprefix = f"{config['sample_out_folder']}/filtered_{noiseratio}_{pllaccept}"
    SeqIO.write(filteredrecords, f"{outprefix}.fasta", "fasta")
    return

if __name__ == "__main__":
    main()
    
    
