import argparse
from raygun.modelv3.raygun import Raygun 
from raygun.modelv3.esmdecoder import DecoderBlock
from raygun.modelv3.loader import RaygunData
from raygun.modelv3.ltraygun import RaygunLightning
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

from E1.modeling import E1ForMaskedLM
from E1.batch_preparer import E1BatchPreparer
from raygun.modelv3.e1_adapter import E1Adapter

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fasta")
    parser.add_argument("valid_fasta")
    parser.add_argument("model_saveloc")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mod", default="8.8M", choices=["2.2M", "4.4M", "8.8M"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--clip", type=float, default=0.00001)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--maxlength", type=int, default=1500)
    parser.add_argument("--minlength", type=int, default=50)
    parser.add_argument("--reclosswt", type=float, default=1)
    parser.add_argument("--replosswt", type=float, default=1)
    parser.add_argument("--celosswt", type=float, default=1)
    parser.add_argument("--log_wandb", default=False, action="store_true")
    parser.add_argument("--reduction", type=int, default=50)
    parser.add_argument("--num_to_save", type=int, default=4)
    parser.add_argument("--data_workers", type=int, default=8)
    parser.add_argument("--per_epoch_val_check", type=int, default=1)
    configs = parser.parse_args()
    return configs.__dict__

def main():
    config = get_args()
    logger.info("Running Raygun training...")

    os.makedirs(config["model_saveloc"], exist_ok=True)

    model_name = "Profluent-Bio/E1-600m"
    e1_model = E1ForMaskedLM.from_pretrained(model_name)
    e1_model.eval()
    batch_preparer = E1BatchPreparer()

    vocab = batch_preparer.tokenizer.get_vocab()
    print("E1 vocab size:", len(vocab))
    print("First 40 tokens:", list(vocab.keys())[:40])
    print("pad_token_id:", getattr(batch_preparer, "pad_token_id", vocab.get("<pad>")))
    print("mask_token_id:", getattr(batch_preparer, "mask_token_id", vocab.get("<mask>")))

    adapter = E1Adapter(
        e1_model=e1_model,
        batch_preparer=batch_preparer,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        use_autocast=True
    )

    esmmodel = adapter
    esmalphabet = adapter

    if config["log_wandb"]:
        wandb_logger = WandbLogger(project="BATCH-TRAINING-RAYGUN-E1-2.2MIL")
    else:
        wandb_logger = None

    wandb_logger = WandbLogger(project="BATCH-TRAINING-RAYGUN-E1-2.2MIL")

    rmodel = Raygun(
        numencoders=12,
        numdecoders=12,
        fixed_esm_batching=True,
        esmmodel=adapter
    )

    rayltmodule = RaygunLightning(
        raygun=rmodel,
        emodel=adapter,
        esmalphabet=getattr(adapter, "token_to_id", None),
        lr=config.get("lr", 1e-4),
        crossentropyloss=config.get("celosswt", 1.0),
        reconstructionloss=config.get("reclosswt", 1.0),
        replicateloss=config.get("replosswt", 1.0),
        log_wandb=config.get("log_wandb", False),
        finetune=False,
        e1=True
    )

    checkpoint = config.get("checkpoint", None)
    if checkpoint is not None:
        with warnings.catch_warnings(record=True):
            rayltmodule = RaygunLightning.load_from_checkpoint(
                checkpoint,
                raygun=rmodel,
                emodel=adapter,
                strict=False,
                e1=True
            )

            for p in rayltmodule.model.encoder.parameters():
                p.requires_grad = True

            for p in rayltmodule.model.decoder.parameters():
                p.requires_grad = True

            rayltmodule.configure_optimizers()

    redlen = config["reduction"]
    rayltmodule.model.encoder.redlength = redlen
    rayltmodule.model.encoder.reduction.reduce_size = redlen

    rayltmodule.traininglog = config["model_saveloc"] + "/error-log.txt"
    rayltmodule.log_wandb = config.get("log_wandb", False)
    rayltmodule.lr = config.get("lr", 1e-3)
    rayltmodule.finetune = False
    rayltmodule.epoch = 0

    traindata = RaygunData(
        fastafile=config["train_fasta"],
        alphabet=esmalphabet,
        batch_preparer=batch_preparer,
        maxlength=config["maxlength"],
        minlength=config["minlength"]
    )

    trainloader = DataLoader(
        traindata,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=traindata.collatefn_with_e1,
        num_workers=config["data_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    validdata = RaygunData(
        fastafile=config["valid_fasta"],
        alphabet=esmalphabet,
        batch_preparer=batch_preparer,
        maxlength=config["maxlength"],
        minlength=config["minlength"]
    )

    validloader = DataLoader(
        validdata,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=validdata.collatefn_with_e1,
        num_workers=config["data_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    chk_callback = ModelCheckpoint(
        monitor="val_blosum_ratio",
        mode="max",
        save_top_k=config["num_to_save"],
        save_weights_only=True,
        dirpath=config["model_saveloc"],
        filename="model-e{epoch:02d}-s{step:06d}-{val_blosum_ratio:.4f}",
        save_on_train_epoch_end=False
    )

    from lightning.pytorch.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[chk_callback, lr_monitor],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        accelerator="gpu",
        val_check_interval=1 / config["per_epoch_val_check"],
        devices=config["devices"],
        precision="bf16",
        strategy="auto",
        max_epochs=150,
        gradient_clip_val=config["clip"],
        gradient_clip_algorithm="norm",
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    rayltmodule.cuda()
    trainer.fit(rayltmodule, trainloader, validloader)

if __name__ == "__main__":
    main()
