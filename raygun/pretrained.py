import torch

def raygun_2_2mil_800M(return_lightning_module = False, return_esmdecoder = False, local=False, localurl=None):
    """
    If return_lightning_model is set to true, it returns the Raygun model wrapped in a Lightning object, making the train operation 
    easier.
    Set it to false by default.
    
    If return_esmdecoder is set to true, then the function will return a model that accepts ESM-2 650M embedding and returns the 
    amino acid logits. 
    """
    if not local:
        raymodel, esmdecoder, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_2_2mil_800M")
    else:
        assert localurl is not None, "Provide the local folder containing the hubconf"
        raymodel, esmdecoder, _ = torch.hub.load(localurl, "pretrained_uniref50_2_2mil_800M", source = "local")
    if not return_lightning_module:
        raymodel = raymodel.model 
        
    if return_esmdecoder:
        return raymodel, esmdecoder
    else:
        return raymodel
    
def raygun_4_4mil_800M(return_lightning_module = False, return_esmdecoder = False, local=False, localurl=None):
    """
    If return_lightning_model is set to true, it returns the Raygun model wrapped in a Lightning object, making the train operation 
    easier.
    Set it to false by default.
    
    If return_esmdecoder is set to true, then the function will return a model that accepts ESM-2 650M embedding and returns the 
    amino acid logits. 
    """
    if not local:
        raymodel, esmdecoder, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_4_4mil_800M")
    else:
        assert localurl is not None, "Provide the local folder containing the hubconf"
        raymodel, esmdecoder, _ = torch.hub.load(localurl, "pretrained_uniref50_4_4mil_800M", source = "local")
    if not return_lightning_module:
        raymodel = raymodel.model 
        
    if return_esmdecoder:
        return raymodel, esmdecoder
    else:
        return raymodel    

    
def raygun_100k_750M(return_esmdecoder = False, local=False, localurl=None):
    if not local:
        raymodel, esmdecoder, hyparams = torch.hub.load('rohitsinghlab/raygun', 
                                                    'pretrained_uniref50_95000_750M')
    else:
        assert localurl is not None, "Provide the local folder containing the hubconf"
        raymodel, esmdecoder, hyparams = torch.hub.load(localurl, 
                                                    'pretrained_uniref50_95000_750M',
                                                       source="local")
        if return_esmdecoder:
            return raymodel, esmdecoder
        else:
            return raymodel