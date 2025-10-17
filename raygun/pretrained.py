import torch 
from raygun.modelv2.raygun import Raygun as RaygunV2
from raygun.modelv2.ltraygun import RaygunLightning
from raygun.modelv2.esmdecoder import DecoderBlock
from esm.pretrained import esm2_t33_650M_UR50D

esmtokens = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}


def load_raymodel(checkpoint, fixed_batching, 
                  num_encoders=12, num_decoders=12):
    esmmodel, esmalph = esm2_t33_650M_UR50D()
    raymodel          = RaygunV2(numencoders=num_encoders, 
                                numdecoders=num_decoders, 
                                fixed_esm_batching=fixed_batching)
    raylightning      = RaygunLightning(raymodel, esmmodel)
    raylightning.load_state_dict(checkpoint["state_dict"],
                                strict=False)
    return raylightning

def raygun_2_2mil_800M(return_lightning_module = False, return_esmdecoder = False, local=False, localurl=None):
    """
    If return_lightning_model is set to true, it returns the Raygun model wrapped in a Lightning object, making the train operation 
    easier.
    Set it to false by default.
    
    If return_esmdecoder is set to true, then the function will return a model that accepts ESM-2 650M embedding and returns the 
    amino acid logits. 
    """
    updatedrayurl = "https://zenodo.org/records/15578824/files/may30-chkpoint-trained-on-4.4m.ckpt?download=1"
    checkpoint    = torch.hub.load_state_dict_from_url(updatedrayurl, progress=True,
                                                   map_location = torch.device("cpu"))
    raymodel      = load_raymodel(checkpoint, fixed_batching=False)
    del checkpoint
    if not return_lightning_module:
        raymodel = raymodel.model 
    return raymodel
    
def raygun_4_4mil_800M(return_lightning_module = False, return_esmdecoder = False, local=False, localurl=None):
    """
    If return_lightning_model is set to true, it returns the Raygun model wrapped in a Lightning object, making the train operation 
    easier.
    Set it to false by default.
    
    If return_esmdecoder is set to true, then the function will return a model that accepts ESM-2 650M embedding and returns the 
    amino acid logits. 
    """
    updatedrayurl = "https://zenodo.org/records/15578824/files/may30-chkpoint-trained-on-4.4m.ckpt?download=1"
    checkpoint    = torch.hub.load_state_dict_from_url(updatedrayurl, progress=True,
                                                   map_location = torch.device("cpu"))
    raymodel      = load_raymodel(checkpoint, fixed_batching=False)
    del checkpoint
    if not return_lightning_module:
        raymodel = raymodel.model 
    return raymodel
    
def raygun_100k_750M(return_esmdecoder = False, localurl=None):
    raymodel, esmdecoder, hyparams = torch.hub.load('rohitsinghlab/raygun', 
                                                'pretrained_uniref50_95000_750M')
    if return_esmdecoder:
        return raymodel, esmdecoder
    else:
        return raymodel
        
def raygun_8_8mil_800M(return_lightning_module = False):
    """
    If return_lightning_model is set to true, it returns the Raygun model wrapped in a Lightning object, making the train operation 
    easier.
    Set it to false by default.
    
    If return_esmdecoder is set to true, then the function will return a model that accepts ESM-2 650M embedding and returns the 
    amino acid logits. 
    """
    updatedrayurl = "https://zenodo.org/records/17253788/files/species_function_aware_sep30_val_blosum_0.9856.ckpt?download=1"
    checkpoint    = torch.hub.load_state_dict_from_url(updatedrayurl, progress=True,
                                                   map_location = torch.device("cpu"))
    raymodel      = load_raymodel(checkpoint, fixed_batching=True)
    del checkpoint
    if not return_lightning_module:
        raymodel = raymodel.model 
    return raymodel