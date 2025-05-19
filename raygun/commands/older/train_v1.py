# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
import sys
from raygun.model.raygun import Raygun 
from raygun.model.esmdecoder import DecoderBlock
from raygun.model.loader import RaygunData
from raygun.train_utils import train
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
import time
import json
from Bio.Align import substitution_matrices
import subprocess
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
        
        assert hasattr(self, "output_model_loc"), "Output save location not specified"
        os.makedirs(self.output_model_loc, exist_ok = True)
        # specify the substitution matrices
        bl = substitution_matrices.load("BLOSUM62")
        self.blosummat = pd.DataFrame(bl, columns = list(bl.alphabet))
        self.blosummat.index = list(bl.alphabet)
        file_handler = logging.FileHandler(f"{self.output_model_loc}/output.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        if not hasattr(self, "finetune"):
            self.finetune = False
            
        if not hasattr(self, "validfasta"):
            self.validfasta = None

        os.makedirs(self.output_model_loc, exist_ok = True)
        # saving the configuration file to the output
        cfgname = cfgfile.split("/")[-1]
        subprocess.call(["cp", cfgfile, f"{self.output_model_loc}/{cfgname}"])
        
        DECODERDIM=1280
        DECODERNHEAD=20
        self.decodermodel = DecoderBlock(DECODERDIM, DECODERNHEAD).to(self.device)
        
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
  
def main():
    logger.info("Running Raygun training/finetuning...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Configuration file")
    config = Config(parser.parse_args().config)
    # Use ESM-2 650M
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()

    # DEFAULT; use the pretrained model
    # if use_pretrained is set to False explicitly,
    # then only the model will be loaded from the checkpoint file
    # or be initialized from scratch
    if hasattr(config, "checkpoint"):
        logger.info(f"Checkpoint found at {config.checkpoint}. Running training/finetuning starting at this point.")
        checkpoint = torch.load(config.checkpoint)
        hyparams = checkpoint["model_hyperparams"]
        config.modelhyperparams = hyparams
        model = Raygun(dim = hyparams["dim"],
                      convkernel = hyparams["convkernel"],
                      numencoders = hyparams["numencoders"],
                      numdecoders = hyparams["numdecoders"],
                      reduction = hyparams["reduction"],
                      nhead = hyparams["nhead"],
                      esm_alphabet = esmalphabet.to_dict()).to(config.device)
        config.update_decodermodel_weights(checkpoint["esmtotokensdecoder"])
        model.load_pretrained(checkpoint)
    else: 
        logger.info(f"Using pre-trained checkpoint.")
        model, esmtotokdecoder, hyparams = torch.hub.load('rohitsinghlab/raygun', 
                                                'pretrained_uniref50_95000_750M')
        del config.decodermodel
        model = model.to(config.device)
        esmtotokdecoder = esmtotokdecoder.to(config.device)
        config.decodermodel = esmtotokdecoder
        config.modelhyperparams = hyparams
        

    # Set finetune to true when want to run the training in the finetune mode
    if not config.finetune:
        paramstotrain = itertools.chain(model.encoder.parameters(),
                                       model.decoder.parameters())
    else:
        paramstotrain = model.decoder.parameters()
            
    optimizer = torch.optim.Adam(paramstotrain, lr = config.lr)
    
    try:
        if "opt_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["opt_state"])
        del checkpoint
    except Exception as e:
        pass
    
    config.get_alphabet(esmalphabet)
    
    traindata = RaygunData(esmmodel,
                        esmalphabet,
                        fastafile = config.trainfasta,
                        saveembedfolder = config.esm2_embedding_saveloc,
                        save   = config.esm2_embedding_saveloc is not None,
                        device = config.device,
                        config = config)
    
    trainloader = DataLoader(traindata, 
                             shuffle = True, 
                            batch_size = 1)
    
    if config.validfasta is not None:
        validdata = RaygunData(esmmodel,
                            esmalphabet,
                            fastafile = config.validfasta,
                            saveembedfolder = config.esm2_embedding_saveloc,
                            save   = config.esm2_embedding_saveloc is not None,
                            device = config.device,
                            config = config)
        validloader = DataLoader(validdata, 
                              shuffle = False,
                              batch_size = 1)
    else:
        validloader = None
    
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.5, 
                                                  end_factor = 1, total_iters = 50, 
                                                 last_epoch=-1)
    
    train(model, trainloader, validloader, optimizer, scheduler, 
         config.epoch, config, config.output_model_loc, config.save_every,
         logger)         
    return

if __name__ == "__main__":
    main()
    
