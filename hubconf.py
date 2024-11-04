import torch 
from raygun.model.raygun import Raygun

esmtokens = {'<cls>': 0,
         '<pad>': 1,
         '<eos>': 2,
         '<unk>': 3,
         'L': 4,
         'A': 5,
         'G': 6,
         'V': 7,
         'S': 8,
         'E': 9,
         'R': 10,
         'T': 11,
         'I': 12,
         'D': 13,
         'P': 14,
         'K': 15,
         'Q': 16,
         'N': 17,
         'F': 18,
         'Y': 19,
         'M': 20,
         'H': 21,
         'W': 22,
         'C': 23,
         'X': 24,
         'B': 25,
         'U': 26,
         'Z': 27,
         'O': 28,
         '.': 29,
         '-': 30,
         '<null_1>': 31,
         '<mask>': 32}

def pretrained_uniref50_95000_750M(pretrained=True, progress=True):
    global esmtokens
    url = "https://zenodo.org/records/14031281/files/raygun-pretrained.sav?download=1"
    
    ## OLDER version "https://zenodo.org/records/13328458/files/raygun-pretrained.sav?download=1". The new link could be loaded into CPU
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url, progress=progress)
        hyparams = checkpoint["model_hyperparams"]
        hyparams["esm_alphabet"] = esmtokens
        model = Raygun(dim = hyparams["dim"],
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