from raygun.modelv2.ltraygun import RaygunLightning
from torch.utils.data import DataLoader
from raygun.modelv2.loader import RaygunData
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import os
import warnings
import torch

def training(ltmodel, esmalphabet, 
             trainfasta, validfasta, outfld, 
             devices=1, clip=0.001, lr=1e-4, 
             epoch=5, batchsize=2, finetune=True,
             delete_checkpoint_after_loading = True):
    os.makedirs(outfld, exist_ok=True)
    ltmodel.lr          = lr
    ltmodel.finetune    = finetune
    ## starting epoch
    ltmodel.epoch       = 0
    ltmodel.traininglog = f"{outfld}/traininglog.txt"
    ltmodel.log_wandb   = False
    
    esmmodel            = ltmodel.esmmodel
    
    ## train loaders
    traindata = RaygunData(fastafile    = trainfasta,
                           alphabet     = esmalphabet,
                           model        = esmmodel)
    trainloader = DataLoader(traindata, 
                             shuffle    = False, 
                             batch_size = batchsize,
                             collate_fn = traindata.collatefn_wo_esm)
    ## validation loaders
    validdata = RaygunData(fastafile    = validfasta,
                           alphabet     = esmalphabet,
                           model        = esmmodel)
    validloader = DataLoader(validdata, 
                            shuffle    = False,
                            batch_size = batchsize, 
                            collate_fn = validdata.collatefn_wo_esm)
    
    chk_callback = ModelCheckpoint(
                        monitor           = "val_blosum_ratio",
                        mode              = "max",
                        save_top_k        = 1, 
                        save_weights_only = True, 
                        dirpath           = outfld,
                        filename          = "model-{epoch:02d}-{step:06d}-{val_blosum_ratio:.4f}",
                        save_on_train_epoch_end = True)

    trainer = L.Trainer(accumulate_grad_batches = 2,
                        callbacks = [chk_callback],
                        accelerator             = "gpu", 
                        devices                 = 1, 
                        max_epochs              = epoch, 
                        gradient_clip_val       = clip,
                        gradient_clip_algorithm = "value")
    
    trainer.fit(ltmodel, 
                trainloader, 
                validloader)
    
    chkptloc = [ckpt for ckpt in Path(outfld).iterdir() 
               if ckpt.suffix == ".ckpt"][0]
    
    # load from the checkpoint
    with warnings.catch_warnings(record=True) as w:
        trained_ltmodel = RaygunLightning.load_from_checkpoint(chkptloc,
                                                              raygun=ltmodel.model,
                                                              esmmodel=ltmodel.esmmodel,
                                                              strict=False)
    
    if delete_checkpoint_after_loading:
        os.remove(chkptloc)
    
    return trained_ltmodel