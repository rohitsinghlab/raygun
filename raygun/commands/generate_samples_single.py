# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
import sys
from raygun.modelv2.raygun import Raygun
from raygun.modelv2.esmdecoder import DecoderBlock
from raygun.modelv2.loader import RaygunData
from raygun.modelv2.ltraygun import RaygunLightning
from raygun.modelv2.training import training
from raygun.pretrained import raygun_2_2mil_800M
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
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("templatefasta", 
                        help = "Template fasta containing a single record. For more than one records, use `generate_samples_multiple.py`")
    parser.add_argument("sample_out_folder", 
                        help = "Output folder")
    parser.add_argument("--minlength", type = int, required = True,
                        help = "Minimum length")
    parser.add_argument("--maxlength", type = int, required = True,
                        help = "Maximum length")
    parser.add_argument("--noiseratio", type = float, default = 1.0,  
                        help = "Noise to introduce during generation")
    parser.add_argument("--num_raygun_samples_to_generate", default = 50, type = float, 
                       help  = "The number of Raygun samples after PLL filtering")
    parser.add_argument("--sample_ratio", default = 10, type = float, 
                       help  = "`#total raygun samples` / `#PLL-filtered raygun samples`")
    parser.add_argument("--randomize_noise", action = "store_true", default = False, 
                       help  = "If true then, in each sample generation, randomly choose a error-ratio between 0 and `noiseratio`")
    parser.add_argument("--device", type = int, default = 0, 
                        help = "GPU device. If CPU, use -1")
    parser.add_argument("--numcycles", type = int, default = 1, 
                       help = "The number of iterative cycles to perform before the final generated sequence")
    parser.add_argument("--penalizerepeats", action = "store_true", default = False, 
                       help = "If true, then penalize the repeats.")
    parser.add_argument("--finetune", action = "store_true", default = False, 
                       help = "If true, then perform finetuning before generation. Only use in rare circumstances where the off-the-shelf sequence identity is relatively small. (<0.9)")
    parser.add_argument("--finetune-trainf", help = "Finetune train fasta file. Needed only when finetune is set to true")
    parser.add_argument("--finetune-validf", help = "Finetune valid fasta file. Needed only when finetune is set to true")
    parser.add_argument("--finetune-epochs", default=10, type=int, help="How many epochs to finetune. Used only when finetune set to true")
    parser.add_argument("--finetune-lr", default=1e-5, type=float, help="Finetune learning rate. Used only when finetune set to true")
    parser.add_argument("--finetune-bsize", default=2, type=int, help="Finetune batch size. Used only when finetune set to true")
    configs = parser.parse_args()
    if configs.device < 0:
        configs.device = "cpu" 
    return configs.__dict__
   
def get_cycles(embedding, finallength, model, nratio,
               numcycles = 0, esmmodel = None, esmbc = None, 
               device = 0):
    batch, seq, dim = embedding.shape
    assert batch == 1, "Batch size should be 1 for generation"
    encoder    = model.encoder(embedding, 
                               noise = nratio)
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
    
def get_model(config, esmmodel, esmalph):
    raymodel = raygun_2_2mil_800M(return_lightning_module=True)
    if config["finetune"]:
        ep     = config["finetune_epochs"]
        tfasta = config["finetune_trainf"]
        vfasta = config["finetune_validf"]
        lr     = config["finetune_lr"]
        bsize  = config["finetune_bsize"]
        outdir = config["sample_out_folder"]
        
        assert config["device"] >= 0, "Finetune set to true, but no GPU provided"
        
        logger.info(f"Finetune set to true. Starting finetuning for {ep} epochs, with lr={lr} on train:{tfasta} and valid:{vfasta}.")
        checkpoint = training(raymodel, esmmodel, esmalph, 
                             tfasta, vfasta, outdir, lr=lr, 
                             epoch=ep, batchsize=bsize)
        raymodel.load_state_dict(checkpoint)
        del checkpoint    
    return raymodel.model
        

def main():
    config = get_params()
    logger.info("Started the Raygun generation process")
    logger.info(f"Penalizerepeats set to {config['penalizerepeats']}.")
    logger.info(f"Length-agnostic PLL filtering activated. Filter ratio: {config['sample_ratio']}")
    logger.info(f"Sample fasta file: {config['templatefasta']}")
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esmmodel              = esmmodel.to(config["device"])
    bc                    = esmalphabet.get_batch_converter()
    
    # load the model 
    raymodel              = get_model(config, esmmodel, esmalphabet)
    raymodel              = raymodel.to(config["device"])
    Path(config["sample_out_folder"]).mkdir(exist_ok=True)
    
    logger.info("Start Raygun sampling: Chooses the first record in the fasta file.")
    preddata = RaygunData(fastafile = config["templatefasta"],
                          alphabet  = esmalphabet,
                          model     = esmmodel,
                          device    = config["device"],
                          no_records = 1)
    print(f"\t\tNo of sequences to generate: {len(preddata)}")
    predloader = DataLoader(preddata, batch_size = 1, shuffle = False,
                           collate_fn = preddata.collatefn)
    
    shrinkeddata = []
    averaged_encoder = 0
    
    noiseratio = config["noiseratio"]
    pllaccept  = config["num_raygun_samples_to_generate"]
    togenerate = int(pllaccept * config["sample_ratio"])
    
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
                length      = np.random.randint(config["minlength"], config["maxlength"]+1)
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
