from collections import defaultdict
import hydra
from esm.pretrained import esm2_t33_650M_UR50D
from raygun.rayflow.loader import RaygunContrastiveDataset
from torch.utils.data import DataLoader
from raygun.pretrained import raygun_4_4mil_800M
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf, DictConfig
import logging
from raygun.modelv2.raygun import Raygun
from raygun.rayflow.rf_ddpm import RayFlowDenoiser
from raygun.rayflow.ltrayflow import RayFlowLightning
import warnings
from datetime import datetime
import math
import torch
torch.autograd.set_detect_anomaly(True)



def cosine_schedule(T, s=0.008):
    steps     = torch.arange(T + 1, dtype=torch.float64)
    f         = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]  
    betas     = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betat     = torch.clamp(betas, 
                       min=1e-8, 
                       max=0.999)
    return betat


# @rank_zero_only
def get_wandb_logger():
    return WandbLogger(project = "Contrastive-Raygun-with-Species-Denoiser")

def construct_dataloader(trainfile, valfile, 
                        esmalph, bsize=3, nworkers=8):
    traindata   = RaygunContrastiveDataset(trainfile, esmalph)
    valdata     = RaygunContrastiveDataset(valfile, esmalph)
    trainloader = DataLoader(traindata,  
                            batch_size=bsize, collate_fn=traindata.collatefn,
                            num_workers=nworkers, shuffle=False, 
                            persistent_workers=True, pin_memory=True)
    valloader   = DataLoader(valdata, 
                            batch_size=bsize, collate_fn=valdata.collatefn,
                            num_workers=nworkers, shuffle=False, 
                            persistent_workers=True, pin_memory=True)
    return trainloader, valloader
    
import sys

@hydra.main(version_base=None, config_path="configs/",
           config_name="config")
def main(config : DictConfig):
    config         = OmegaConf.to_container(config, resolve=True)
    TRLMDB         = config["TRAINLMDB"]
    VALMDB         = config["VALMDB"]
    EPOCH          = config["EPOCH"]
    devices        = config["DEVICES"]
    MODELFLD       = config["MODELFLD"]
    BSIZE          = config["BATCH_SIZE"]
    STANDARDLOSSWT = config["STANDARDLOSSWT"]
    FUNCTION_DIM   = config["FUNCTION_DIM"]
    LR             = config["LR"]
    NSPECIES       = config["NSPECIES"]
    NWORKERS       = config.get("DATAWORKERS", 8 )
    FIXED_DIM      = config.get("FIXED_DIM"  , 50)
    RAY_ENCODER    = config.get("RAY_ENCODER", 12)
    RAY_DECODER    = config.get("RAY_DECODER", 12)
    NHEAD          = config.get("NHEAD"      , 20)
    
    logging.info("Loading ESM-2 model and DataLoaders...")
    esmmodel, esmalph      = esm2_t33_650M_UR50D()
    trainloader, valloader = construct_dataloader(TRLMDB, VALMDB, 
                                                  esmalph, bsize=BSIZE,
                                                  nworkers=NWORKERS)
    TIME                   = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    
    logging.info("Loading Raygun...")

    # load the Raygun model 
    raymodel                = Raygun(numencoders       = RAY_ENCODER, 
                                     numdecoders       = RAY_DECODER,
                                     reduction         = FIXED_DIM,
                                     fixed_esm_batching=True)

    # Rayflow DDPM
    CSCHEDULE      = config.get("COSINE_SCHEDULE"           , 1000)
    FLOW_N_BLOCK   = config.get("RAYFLOW_BLOCK"             , 5   )
    FLOW_T_H_DIM   = config.get("RAYFLOW_TIME_HIDDEN_DIM"   , 1280)
    FLOW_SP_E_DIM  = config.get("RAYFLOW_SPECIES_EMB_DIM"   , 1280)
    FLOW_SP_H_DIM  = config.get("RAYFLOW_SPECIES_HIDDEN_DIM", 1280)
    FLOW_EMBED_DIM = config.get("RAYFLOW_EMBED_DIM"         , 1280)
    FLOW_CONV      = config.get("FLOW_CONVSIZE"             , 7   )
    
    FLOW_SP_ST_DIM = (FLOW_EMBED_DIM-FUNCTION_DIM)*FIXED_DIM
    beta_schedule  = cosine_schedule(T=CSCHEDULE)
    
    rayddpm        = RayFlowDenoiser(beta_schedule, 
                                     no_flowblock=FLOW_N_BLOCK, 
                                     embed_dim=FLOW_EMBED_DIM, 
                                     t_hidden_dim=FLOW_T_H_DIM, 
                                     sp_emb_dim=FLOW_SP_E_DIM, 
                                     sp_st_dim=FLOW_SP_ST_DIM, 
                                     fixed_dim=FIXED_DIM, 
                                     convkernel=FLOW_CONV,
                                     max_species=NSPECIES,
                                     nhead=NHEAD)
    
    checkpoint        = config["checkpoint"]
    RAYGUN_CELOSS     = config.get("RAYGUN_CE_LOSS"    , 1.  )
    RAYGUN_RECLOSS    = config.get("RAYGUN_REC_LOSS"   , 1.  )
    RAYGUN_REPLOSS    = config.get("RAYGUN_REP_LOSS"   , 1.  )
    RAYGUN_LOSSWT     = config.get("RAYGUN_STLOSS_WT"  , 1.  )
    RAY_CONTRASTIVEWT = config.get("RAY_CONTRASTIVE_WT", 0.25)
    RAY_DENOISINGWT   = config.get("RAY_DENOISING_WT"  , 1.  )
    
    
    with warnings.catch_warnings(record=True) as w:
        rayltmodule   = RayFlowLightning.load_from_checkpoint(checkpoint, 
                                                              raygun=raymodel,
                                                              esmmodel=esmmodel,
                                                              rfdenoiser=rayddpm,
                                                              no_species=NSPECIES,
                                                              lr=LR,
                                                              standardlosswt=STANDARDLOSSWT,
                                                              contrastivewt=RAY_CONTRASTIVEWT,
                                                              denoisewt=RAY_DENOISINGWT,
                                                              function_dim=FUNCTION_DIM,
                                                              finetune=False,  
                                                              traininglog=f"{MODELFLD}/training-logs-{TIME}.txt",
                                                              strict=False)
    logging.info(f"Using pre-trained checkpoint. Esmdecoder batching {rayltmodule.model.esmdecoder.fixed_batching}")
    
    
    logging.info("Launching the Trainer...")
    logger      = get_wandb_logger()
    
    chk_callback = ModelCheckpoint(monitor            = "val_blosum_score",
                               mode                   = "max",
                               save_top_k             = 10, 
                               save_weights_only      = True, 
                               dirpath                = f"{MODELFLD}/",
                               filename               = "EP-{epoch:02d}-{step:06d}-{val_blosum_score:.5f}-{val_blosum_ratio:.5f}",
                               save_on_train_epoch_end=False)
    
    trainer     = L.Trainer(logger                    = logger,
                           accelerator                = "gpu", 
                           callbacks                  = [chk_callback],
                           val_check_interval         = 0.25,
                           check_val_every_n_epoch    = 1, 
                           devices                    = devices, 
                           max_epochs                 = EPOCH, 
                           strategy                   = "ddp_find_unused_parameters_true",
                           log_every_n_steps          = 5,)
    
    trainer.fit(rayltmodule, trainloader, valloader)

if __name__ == "__main__":
    main()
    