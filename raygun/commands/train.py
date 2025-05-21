# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
from raygun.modelv2.raygun import Raygun 
from raygun.modelv2.esmdecoder import DecoderBlock
from raygun.modelv2.loader import RaygunData
from raygun.modelv2.ltraygun import RaygunLightning
from raygun.pretrained import raygun_2_2mil_800M
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
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from datetime import datetime
from pathlib import Path

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@hydra.main(config_path="configs/", config_name="train",
            version_base = None)
def main(config: DictConfig):
    logger.info("Running Raygun training...")
    config = OmegaConf.to_container(config, resolve=True)

    # create model and embedding folders
    os.makedirs(config["model_saveloc"], exist_ok = True)
    if config["esm2_embedding_saveloc"] is not None:
        os.makedirs(config["model_saveloc"], exist_ok = True)

    # Use ESM-2 650M
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esmmodel              = esmmodel.to(0)
    esmmodel.eval()

    if config["log_wandb"]:
        wandb_logger = WandbLogger(project = "BATCH-TRAINING-RAYGUN")
    else:
        wandb_logger = None

    logger.info(f"Using pre-trained checkpoint.")
    # load the model 
    rayltmodule             = raygun_2_2mil_800M(return_lightning_module=True)
    
    if "checkpoint" in config and config["checkpoint"] is not None:
        ckptpath   = Path(config["checkpoint"])
        checkpoint = torch.load(ckptpath, weights_only = True)
        rayltmodule.load_state_dict(checkpoint["state_dict"])

    rayltmodule.traininglog = config["model_saveloc"] + "/error-log.txt"
    rayltmodule.log_wandb   = config["log_wandb"]
    rayltmodule.lr          = config["lr"]
    rayltmodule.finetune    = False
    rayltmodule.epoch       = 0
    
    ## train and validation loaders
    traindata = RaygunData(fastafile = config["trainfasta"],
                           alphabet  = esmalphabet,
                           model     = esmmodel, 
                           device    = 0)
    trainloader = DataLoader(traindata, 
                             shuffle = True, 
                             batch_size = config["batch_size"],
                             collate_fn = traindata.collatefn)
    validdata = RaygunData(fastafile = config["validfasta"],
                           alphabet  = esmalphabet,
                           model     = esmmodel,
                           device    = 0)
    validloader = DataLoader(validdata, 
                            shuffle = False,
                            batch_size = config["batch_size"], 
                            collate_fn = validdata.collatefn)
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
                        val_check_interval=0.25,
                        devices=config["devices"], strategy="ddp",
                        max_epochs=config["epoch"], 
                        gradient_clip_val = config["clip"],
                        gradient_clip_algorithm = "value")
    
    trainer.fit(rayltmodule, trainloader, 
                validloader)
    return 

if __name__ == "__main__":
    main()
    
