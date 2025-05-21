# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
import sys
from raygun.old_modelv1.raygun import Raygun
from raygun.train_utils import train
from raygun.old_modelv1.esmdecoder import DecoderBlock
from raygun.old_modelv1.loader import RaygunData
from raygun.pll import get_PLL, penalizerepeats
import yaml
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class Config:
    def __init__(self, cfgfile):
        with open(cfgfile, 'r') as file:
            configdict = yaml.safe_load(file)
        self.__dict__.update(configdict)
        
        assert hasattr(self, "sample_out_folder"), "Output sampling folder not provided"
        assert hasattr(self, "output_file_identifier"), "Please provide a name to the sampling task in the `output_file_identifier` option"
        os.makedirs(self.sample_out_folder, exist_ok = True)
        if self.finetune:
            assert (hasattr(self, "finetune_epoch") and hasattr(self, "finetunetrain")), "number of epochs or train fasta not specified"
            assert hasattr(self, "finetuned_model_loc"), "Output finetuned model location not provided"
            assert hasattr(self, "finetune_lr"), "The finetuning learning rate not provided"
            if not hasattr(self, "finetune_save_every"):
                self.finetune_save_every = 1
            os.makedirs(self.finetuned_model_loc, exist_ok = True)
        
        bl = substitution_matrices.load("BLOSUM62")
        self.blosummat = pd.DataFrame(bl, columns = list(bl.alphabet))
        self.blosummat.index = list(bl.alphabet)
            
        subprocess.call(["cp", cfgfile, f"{self.sample_out_folder}/config.yaml"])
        file_handler = logging.FileHandler(f"{self.sample_out_folder}/sampling.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # set penalizerepeats on by default
        if not hasattr(self, "penalizerepeats"):
            self.penalizerepeats = True
        
        if hasattr(self, "noiseratio"):
            nratio = (self.noiseratio if self.noiseratio is not None 
                      else 0)
        else:
            nratio = 0
        
        assert hasattr(self, "lengthinfo"), "Length json file not specified"
        with open(f"{self.lengthinfo}", "r") as jf:
            lengthjs = json.load(jf)
        protstosample    = set([rec.id for rec in 
                            SeqIO.parse(self.templatefasta, "fasta")])
        protswithleninfo = set(lengthjs.keys())
        assert len(protstosample.intersection(protswithleninfo)) == len(protstosample), "Length info not provided for all the proteins to sample"
        self.minlength = {}
        self.maxlength = {}
        for k, v in lengthjs.items():
            assert len(v) == 2, f"Malformed length info for protein: {k}"
            lmin, lmax = v
            assert lmin <= lmax, f"First entry should be smaller than or equal to the second entry"
            self.minlength[k] = lmin
            self.maxlength[k] = lmax
        
        assert hasattr(self, "num_raygun_samples_to_generate"), "Provide the number of raygun samples to generate"
        
        if not hasattr(self, "filter_ratio_with_pll"):
            self.filter_ratio_with_pll = 0.9 # filter out 90 percentage of generated candidates by default
        else:
            assert(self.filter_ratio_with_pll < 1.0), "the ratio of pll filtering should be less than 1"
        self.totalgenerated = int(self.num_raygun_samples_to_generate / (1 - self.filter_ratio_with_pll)) 
        
        if not hasattr(self, "numcycles"):
            self.numcycles = 1
        
        # output directory for the ESM-2 embeddings
        if not hasattr(self, "embed"):
            self.embed = None
        
        DECODERDIM=1280
        DECODERNHEAD=20
        self.decodermodel = DecoderBlock(DECODERDIM, 
                                        DECODERNHEAD).to(self.device)
    
    def update_decodermodel_weights(self, weights):
        self.decodermodel.load_state_dict(weights)
        return
    
    def get_alphabet(self, alphabet):
        alphtokdict = alphabet.to_dict()
        self.toktoalphdict = {k:i for i, k in alphtokdict.items()}
        return
    
    def convert_tokens_to_alph(self, token):
        alphabets = []
        for tok in token:
            alphabets.append(self.toktoalphdict[tok])
        return alphabets
    
    def get_blosum_score(self, embedding, true_token):
        with torch.no_grad():
            true_alph   = self.convert_tokens_to_alph(true_token.flatten().cpu().numpy())
            logits      = self.decodermodel(embedding)
            pred_tokens = torch.argmax(logits, dim = -1).flatten().cpu().numpy()
            pred_alph   = self.convert_tokens_to_alph(pred_tokens)
        return self.compute_blosum_score(true_alph, 
                                        pred_alph)
    
    def compute_blosum_score(self, true, predicted):
        blosum_max  = 0
        blosum_curr = 0
        for p, q in zip(true, predicted):
            try:
                blosum_c_score = self.blosummat.loc[p.upper(), 
                                                    q.upper()] # if no p and q, this triggers exception
                blosum_max += self.blosummat.loc[p.upper(), 
                                                 p.upper()]
                blosum_curr += blosum_c_score
            except Exception as e:
                continue
        return blosum_curr, blosum_curr / blosum_max
    
            
def get_cycles(embedding, finallength, model, config, nratio,
              numcycles = 0, esmmodel = None, esmbc = None):
    encoder = model.encoder(embedding.to(config.device), 
                            error_c = nratio)
    changedseq = model.shrinkwithencoder(encoder, finallength)
    if numcycles == 0: return changedseq
    assert esmmodel is not None and esmbc is not None
    esmmodel = esmmodel.to(config.device)
    with torch.no_grad():
        esmmodel.eval()
        model.eval()
        for cycle in range(numcycles):
            data = [("prot", changedseq)]
            _, _, tokens = esmbc(data)
            tokens = tokens.to(config.device)
            embedcycle = esmmodel(tokens, repr_layers = [33],
                                 return_contacts = False)["representations"][33][:, 1:-1, :]
            embedcycle = model.encoder(embedcycle) # do not add noise here
            changedseq = model.shrinkwithencoder(embedcycle, finallength)
    return changedseq


def finetune(esmmodel, esmalphabet, model, 
             trainfasta, config, 
             logger, validfasta = None, do_validations = False):
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr = config.finetune_lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.5, 
                                                  end_factor = 1, total_iters = 50, 
                                                 last_epoch=-1)
    traindata = RaygunData(esmmodel,
                        esmalphabet,
                        fastafile = trainfasta,
                        saveembedfolder = config.embed,
                        device = config.device,
                        config = config)

    trainloader = DataLoader(traindata, 
                         shuffle = True, 
                        batch_size = 1)

    if (validfasta is not None) and do_validations:
        validdata = RaygunData(esmmodel,
                            esmalphabet,
                            fastafile = validfasta,
                            saveembedfolder = config.embed,
                            device = config.device,
                            config = config)
        validloader = DataLoader(validdata, 
                              shuffle = False,
                              batch_size = 1)
    else:
        validloader = None
    train(model, trainloader, validloader, optimizer, 
          scheduler, config.finetune_epoch,
          config, config.finetuned_model_loc, config.finetune_save_every,
          logger)
    del optimizer, scheduler, trainloader, traindata
    if validloader is not None:
        del validdata, validloader
    return model
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Configuration file")
    config = Config(parser.parse_args().config)
    logger.info("Started the Raygun generation process")
    logger.info(f"Finetuning set to {config.finetune}. {'' if config.finetune else 'We strongly recommend to finetune the the pretrained Raygun model before using it THE FIRST TIME time for template-based protein generation. For later use, THE SAVED CHECKPOINTS can be used, setting finetune to False'}")
    logger.info(f"Penalizerepeats set to {config.penalizerepeats}.")
    logger.info(f"Length-agnostic PLL filtering activated. Filter ratio: {config.filter_ratio_with_pll}")
    logger.info(f"Sample fasta file: {config.templatefasta}")
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    # if there is already a finetuned model, use that by setting the model URL in `finetuned_model_loc`.
    if hasattr(config, "finetuned_model_checkpoint"):
        logger.info(f"Loading the checkpoint from {config.finetuned_model_checkpoint}.")
        checkpoint = torch.load(config.finetuned_model_checkpoint, 
                                map_location = torch.device(config.device))
        hyparams = checkpoint["model_hyperparams"]
        model = Raygun(dim = hyparams["dim"],
                      convkernel = hyparams["convkernel"],
                      numencoders = hyparams["numencoders"],
                      numdecoders = hyparams["numdecoders"],
                      reduction = hyparams["reduction"],
                      nhead = hyparams["nhead"],
                      esm_alphabet = esmalphabet.to_dict()).to(config.device)
        config.update_decodermodel_weights(checkpoint["esmtotokensdecoder"])
        model.load_pretrained(checkpoint)
    # use the pretrained model from zenodo
    else: 
        logger.info(f"Loading the pretrained Raygun model")
        model, esmtotokdecoder, hypparams = torch.hub.load('rohitsinghlab/raygun', 
                                                'pretrained_uniref50_95000_750M')
        del config.decodermodel
        model = model.to(config.device)
        esmtotokdecoder = esmtotokdecoder.to(config.device)
        config.decodermodel = esmtotokdecoder
        config.modelhyperparams = hypparams
    bc = esmalphabet.get_batch_converter()
    config.get_alphabet(esmalphabet)
    
    if config.finetune:
        logger.info(f"Started the finetuning process for epoch {config.finetune_epoch}")
        logger.info(f"Finetuning on the fasta file: {config.finetunetrain}")
        # if finetune validation not provided then use the trainset as validation set
        if not hasattr(config, "finetunevalid"):
            config.finetunevalid = None
        finetune(esmmodel, esmalphabet, model, 
                 config.finetunetrain, config, 
                 logger, config.finetunevalid,
                 do_validations = config.finetunevalid is not None)
    logger.info("Start Raygun sampling:")
    preddata = RaygunData(esmmodel,
                          esmalphabet,
                          config.templatefasta,
                          saveembedfolder = None, 
                          prediction = True,
                          device = config.device,
                          config = config)
    
    shrinkeddata = []
    averaged_encoder = 0
    
    noiseratio = config.noiseratio
    togenerate = config.totalgenerated
    pllaccept  = config.num_raygun_samples_to_generate
    
    records = []
    outprefix = f"{config.sample_out_folder}/unfiltered_{config.output_file_identifier}_{noiseratio}_{togenerate}"

    logging.info("Raygun sampling started.")
    nameassignment = {}
    with torch.no_grad():
        model.eval()
        for name, embed in preddata:
            embed = embed.unsqueeze(0)
            for h in tqdm(range(togenerate)):
                nratio      = (noiseratio if (not config.randomize_noise) else 
                               random.random() * noiseratio)
                length      = np.random.randint(config.minlength[name], config.maxlength[name]+1)
                changedseq = get_cycles(embed, length, model, 
                                        config, nratio,
                                        numcycles = config.numcycles,
                                        esmmodel = esmmodel, esmbc = bc)
                genname = f"{name}_i_{h}_l_{length}_n_{nratio}"
                nameassignment[genname] = name
                record = SeqRecord(Seq(changedseq),
                                   id = genname,
                                   description = f"noise ratio added: {nratio}")
                records.append(record)
    SeqIO.write(records, f"{outprefix}.fasta", "fasta")

    del model
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
        if config.penalizerepeats:
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
    
    outprefix = f"{config.sample_out_folder}/filtered_{config.output_file_identifier}_{noiseratio}_{pllaccept}"
    SeqIO.write(filteredrecords, f"{outprefix}.fasta", "fasta")
    return

if __name__ == "__main__":
    main()
    
    
