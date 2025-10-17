import torch 
from raygun.modelv2.raygun import Raygun as RaygunV2
from raygun.modelv2.ltraygun import RaygunLightning
from raygun.modelv2.esmdecoder import DecoderBlock
from esm.pretrained import esm2_t33_650M_UR50D

esmtokens = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

def pretrained_uniref50_95000_750M(pretrained=True, progress=True):
    global esmtokens
    
    from raygun.old_modelv1.raygun import Raygun as RaygunV1
    url = "https://zenodo.org/records/14031281/files/raygun-pretrained.sav?download=1"
    
    ## OLDER version "https://zenodo.org/records/13328458/files/raygun-pretrained.sav?download=1". The new link could be loaded into CPU
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url, progress=progress,
                                                       map_location = torch.device("cpu"))
        hyparams = checkpoint["model_hyperparams"]
        hyparams["esm_alphabet"] = esmtokens
        model = RaygunV1(dim = hyparams["dim"],
                      convkernel = hyparams["convkernel"],
                      numencoders = hyparams["numencoders"],
                      numdecoders = hyparams["numdecoders"],
                      reduction = hyparams["reduction"],
                      nhead = hyparams["nhead"],
                      esm_alphabet = esmtokens)
        model.load_pretrained(checkpoint)
        esmembtotokdecoder = model.esmdecoder
        del checkpoint["model_state"]
    return model, esmembtotokdecoder, hyparams


def pretrained_uniref50_2_2mil_800M(pretrained=True, progress=True):
    updatedrayurl = "https://zenodo.org/records/15447158/files/model-may-16.ckpt?download=1"
    esmdecoderurl = "https://zenodo.org/records/15447367/files/esm-decoder.sav?download=1"
    if pretrained:
        model         = RaygunV2(numencoders = 12, 
                              numdecoders  = 12)
        hyparams      = {"numencoders" : 12, 
                         "numdecoders" : 12,
                         "dim"         : 1280,
                         "dropout"     : 0.1,
                         "reduction"   : 50, 
                         "activation"  : "gelu"}
        rayltmodule      = RaygunLightning(model)
        checkpoint    = torch.hub.load_state_dict_from_url(updatedrayurl, progress=progress,
                                                       map_location = torch.device("cpu"))
        rayltmodule.load_state_dict(checkpoint["state_dict"])
        esmdecoder    = DecoderBlock(dim = 1280, 
                                     nhead = 20)
        deccheckpoint = torch.hub.load_state_dict_from_url(esmdecoderurl, progress=progress,
                                                       map_location = torch.device("cpu"))
        
        esmdecoder.load_state_dict(deccheckpoint["model_state"])
        del deccheckpoint, checkpoint
    return rayltmodule, esmdecoder, hyparams

def pretrained_uniref50_4_4mil_800M(pretrained=True, progress=True):
    updatedrayurl = "https://zenodo.org/records/15578824/files/may30-chkpoint-trained-on-4.4m.ckpt?download=1"
    esmdecoderurl = "https://zenodo.org/records/15447367/files/esm-decoder.sav?download=1"
    if pretrained:
        model         = RaygunV2(numencoders = 12, 
                              numdecoders  = 12)
        hyparams      = {"numencoders" : 12, 
                         "numdecoders" : 12,
                         "dim"         : 1280,
                         "dropout"     : 0.1,
                         "reduction"   : 50, 
                         "activation"  : "gelu"}
        rayltmodule      = RaygunLightning(model)
        checkpoint    = torch.hub.load_state_dict_from_url(updatedrayurl, progress=progress,
                                                       map_location = torch.device("cpu"))
        rayltmodule.load_state_dict(checkpoint["state_dict"])
        esmdecoder    = DecoderBlock(dim = 1280, 
                                     nhead = 20)
        deccheckpoint = torch.hub.load_state_dict_from_url(esmdecoderurl, progress=progress,
                                                       map_location = torch.device("cpu"))
        
        esmdecoder.load_state_dict(deccheckpoint["model_state"])
        del deccheckpoint, checkpoint
    return rayltmodule, esmdecoder, hyparams