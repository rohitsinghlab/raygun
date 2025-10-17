# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
from raygun.modelv2.raygun import Raygun 
from raygun.modelv2.esmdecoder import DecoderBlock
from raygun.modelv2.loader import RaygunData
from raygun.modelv2.ltraygun import RaygunLightning
import raygun.pretrained as pretrained 
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
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from datetime import datetime
from pathlib import Path
import warnings
import argparse

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fasta", help="Train fasta file")
    parser.add_argument("valid_fasta", help="Validation fasta file")
    parser.add_argument("model_saveloc", help="Directory for saving the model")
    parser.add_argument("--checkpoint", default=None, help="Load from previous checkpoint if it exists")
    parser.add_argument("--mod", default="8.8M", choices=["2.2M", "4.4M", "8.8M"], 
                        help="If the previously trained checkpoint does not exist, it starts training from one of three saved checkpoints in zenodo")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices. Default: 1")
    parser.add_argument("--batch_size", type=int, default=4, help="The target batch size. For A100, batch_size=5 generally works.")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--clip", type=float, default=0.00001, help="Maximum gradient clip")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="How many gradient accumulation steps before backward?")
    parser.add_argument("--maxlength", type=int, default=1500, help="Maximum seq length")
    parser.add_argument("--minlength", type=int, default=50, help="Minimum seq length")
    parser.add_argument("--reclosswt", type=float, default=1, help="Reconstruction loss contribution")
    parser.add_argument("--replosswt", type=float, default=1, help="Repetition loss contribution")
    parser.add_argument("--celosswt", type=float, default=1, help="Cross Entropy loss contribution")
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Logging with Wandb enabled")
    parser.add_argument("--reduction", type=int, default=50, help="The fixed length reduced dimension")
    parser.add_argument("--num_to_save", type=int, default=4, help="How many checkpoints to save in each run")
    parser.add_argument("--data_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--per_epoch_val_check", type=int, default=1, help="How many times do we want to do validations per epoch")
    configs = parser.parse_args()
    return configs.__dict__

def main():
    config   = get_args()
    logger.info("Running Raygun training...")

    # create model and embedding folders
    os.makedirs(config["model_saveloc"], exist_ok = True)

    # Use ESM-2 650M
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esmmodel.eval()

    if config["log_wandb"]:
        wandb_logger = WandbLogger(project = "BATCH-TRAINING-RAYGUN")
    else:
        wandb_logger = None
    
    # load the model 
    logger.info(f"Initial model loading setup...")
    checkpoint              = config.get("checkpoint", None)
    if checkpoint is not None:
        numencoders= config.get("numencoders", 12)
        numdecoders= config.get("numdecoders", 12)
        rmodel     = Raygun(numencoders=numencoders,
                            numdecoders=numdecoders, 
                            fixed_esm_batching=True)
        with warnings.catch_warnings(record=True) as w:
            rayltmodule = RaygunLightning.load_from_checkpoint(checkpoint, 
                                                               raygun=rmodel, 
                                                               esmmodel=esmmodel,
                                                               strict=False)
    else: 
        mod        = config.get("pretrained_version", "8.8")
        loadfunc   = (pretrained.raygun_2_2mil_800M if mod=="2.2" else
                     (pretrained.raygun_4_4mil_800M if mod=="4.4" else
                      pretrained.raygun_8_8mil_800M))
        rayltmodule= loadfunc(return_lightning_module=True)
        
    logger.info(f"Using pre-trained checkpoint.")

    rayltmodule.traininglog = config["model_saveloc"] + "/error-log.txt"
    rayltmodule.log_wandb   = config.get("log_wandb", False)
    rayltmodule.lr          = config.get("lr", 1e-3)
    rayltmodule.finetune    = False
    rayltmodule.epoch       = 0
    
    # if the reduction fixed length is to be changed
    redlen                                          = config["reduction"]
    logger.info(f"Setting reduction length to {redlen}")
    rayltmodule.model.encoder.redlength             = redlen
    rayltmodule.model.encoder.reduction.reduce_size = redlen
    
    
    ## train and validation loaders
    traindata = RaygunData(fastafile = config["train_fasta"],
                           alphabet  = esmalphabet,
                           maxlength = config["maxlength"],
                           minlength = config["minlength"])
    
    trainloader = DataLoader(traindata, 
                             shuffle = True, 
                             batch_size = config["batch_size"],
                             collate_fn = traindata.collatefn_wo_esm,
                             num_workers=config["data_workers"],
                             pin_memory=True, 
                             persistent_workers=True, 
                             prefetch_factor=4)
    
    validdata = RaygunData(fastafile = config["valid_fasta"],
                           alphabet  = esmalphabet,
                           maxlength = config["maxlength"],
                           minlength = config["minlength"])
    
    validloader = DataLoader(validdata, 
                             shuffle = False,
                             batch_size = config["batch_size"], 
                             collate_fn = validdata.collatefn_wo_esm,
                             num_workers=config["data_workers"],
                             pin_memory=True, 
                             persistent_workers=True, 
                             prefetch_factor=4)
    # Start the training
    
    ## checkpoint
    chk_callback = ModelCheckpoint(
                        monitor = "val_blosum_ratio",
                        mode    = "max",
                        save_top_k = config["num_to_save"], 
                        save_weights_only = True, 
                        dirpath = config["model_saveloc"],
                        filename = "model-e{epoch:02d}-s{step:06d}-{val_blosum_ratio:.4f}",
                        save_on_train_epoch_end=False
                    )
    
    trainer = L.Trainer(logger = wandb_logger, 
                        callbacks = [chk_callback],
                        accumulate_grad_batches=config["accumulate_grad_batches"],
                        accelerator="gpu", 
                        val_check_interval=1/config["per_epoch_val_check"],
                        devices=config["devices"], 
                        strategy="ddp_find_unused_parameters_true",
                        max_epochs=config["epoch"], 
                        gradient_clip_val = config["clip"],
                        gradient_clip_algorithm = "value")
    
    trainer.fit(rayltmodule, trainloader, 
                validloader)
    return 

if __name__ == "__main__":
    main()
    
