# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
   
def train_epoch(model, loader, optimizer, epoch, config):
    model.train()
    losses = []
    for i, batch in enumerate(tqdm(loader, desc = f"Training epoch {epoch}")):
        tokens, e = batch
        tokens = tokens[:, 1:-1]
        if e.shape[1] <= config.minallowedlength:
            continue
        optimizer.zero_grad()
        e = e.to(config.device)
        tloss = 0
        
        toprint = {}
        
        # use crossentropy loss
        if config.usecrossentropyloss:
            result, mem, crossloss = model(e, tokens.to(config.device))
            tloss = tloss + config.crossentropylossratio * crossloss
            toprint["training cross-entropyloss"] = crossloss.item()
        else:
            result, mem = model(e)
            
        # reconstruction loss
        if config.usereconstructionloss:
            reconstructloss = F.mse_loss(result, e)
            tloss = tloss + config.reconstructionlossratio * reconstructloss
            toprint["training reconstruction loss"] = reconstructloss.item()
            
        # replicate loss
        if config.usereplicateloss:
            newlength  = torch.randint(config.minallowedlength, e.shape[1], [1])[0]
            decodedemb = model.decode(mem, newlength)
            replicateloss = F.mse_loss(mem, model.encoder(decodedemb))
            tloss = tloss + config.replicatelossratio * replicateloss
            toprint["training replicate loss"] = replicateloss.item()
        # crossentropyloss
        
        tloss.backward()
        toprint["total training loss"] = tloss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip if hasattr(config, "clip") else 0.0001)
        optimizer.step()
        
        blosum_curr, blosum_curr_ratio = config.get_blosum_score(result.detach(),
                                                                 tokens)
        toprint["training blosum curr ratio"] = blosum_curr_ratio
        toprint["training blosum curr score"] = blosum_curr

        losses.append(toprint)
    return losses
    
def test_epoch(model, loader, epoch, config):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc = f"Test epoch {epoch}")):
            tokens, e = batch
            tokens = tokens[:, 1:-1]
            if e.shape[1] <= config.minallowedlength:
                continue
            e = e.to(config.device)
            result, mem = model(e)
            
            toprint = {}
            # reconstruction loss
            loss = F.mse_loss(result, e)
            toprint["testing reconstruction loss"] = loss.item()
            
            # replicate loss
            newlength  = torch.randint(config.minallowedlength, e.shape[1], [1])[0]
            decodedemb = model.decoder(mem, newlength)
            lossf = F.mse_loss(mem, model.encoder(decodedemb))
            toprint["testing replicate error"] = lossf.item()
            toprint["testing total error"] = loss.item() + lossf.item()
            
            blosum_curr, blosum_curr_ratio = config.get_blosum_score(result,
                                                                     tokens)
            toprint["testing blosum curr ratio"] = blosum_curr
            toprint["testing blosum curr score"] = blosum_curr_ratio

            losses.append(toprint)
    return losses

def train(model, trainloader, validloader, optimizer, 
          scheduler, n_epochs, config, output_loc, save_every = 1, logger = None):
    for e in range(n_epochs):
        trainloss = train_epoch(model, trainloader, optimizer,
                                e, config)
        trlossmean = pd.DataFrame(trainloss).mean(axis = 0)
        trlossmean = {f"per-epoch {f}": g for f, g in trlossmean.items()}

        trprint = f"Train Epoch {e} complete ---"
        for key, val in trlossmean.items():
            trprint += f"| {key}: {val} |"    
        if logger is not None:
            logger.info(trprint)

        if validloader is not None:
            validloss = test_epoch(model, validloader, e, config)
            validlossmean = pd.DataFrame(validloss).mean(axis = 0)
            validlossmean = {f"per-epoch {f}": g for f, g in validlossmean.items()}
            teprint = f"Valid Epoch {e} complete ---"
            for key, val in validlossmean.items():
                teprint += f"| {key}: {val} |"
            if logger is not None:
                logger.info(teprint)

        # optimizer.step()
        scheduler.step()
        if (e+1) % save_every == 0 or (e == n_epochs - 1):
            savedict = {
                            "model_state"      : model.state_dict(),
                            "trainloss"        : trainloss, 
                            "model_hyperparams": config.modelhyperparams,
                            "esmtotokensdecoder" : model.esmdecoder.state_dict()
                        }
            if validloader is not None:
                savedict["validloss"] = validloss
            if config.saveoptimizerstate:
                savedict["opt_state"] = optimizer.state_dict()
            if logger is not None:
                logger.info(f"Saving to {output_loc}/epoch_{e+1}.sav")
            torch.save(savedict, f"{output_loc}/epoch_{e+1}.sav")
            del savedict