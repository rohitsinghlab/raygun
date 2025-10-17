import warnings
from raygun.modelv2.training import training
from raygun.pretrained import raygun_8_8mil_800M
from esm.pretrained import esm2_t33_650M_UR50D
import torch
import shutil
import os

TRAINFASTA = "data/fastas/ATLAS_valid.fasta"
VALIDFASTA = "data/fastas/ATLAS_valid.fasta"


def test_raymodel_finetune():
    if not torch.cuda.is_available():
        return
    error        = False
    try:
        raymodel = raygun_8_8mil_800M(return_lightning_module=True)
        _, alph  = esm2_t33_650M_UR50D()
        raymodel = training(raymodel, alph, 
                           TRAINFASTA, VALIDFASTA, "out/", 
                           epoch=1)
    except Exception as e:
        error    = True
        exception= e
        
    if os.path.exists("out/"):
        shutil.rmtree("out/")
    
    if error: 
        raise Exception(f"Training failed... {exception}")
    return

