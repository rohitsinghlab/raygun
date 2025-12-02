# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun

import lightning as L
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import logging
from Bio.Align import substitution_matrices
import numpy as np
import itertools
from einops import rearrange
import torch.nn as nn
import parasail

MINALLOWEDLENGTH=50

def pairwise(seq1, seq2):
    out = parasail.nw_trace_diag_16(seq1, seq2, 
                                    1, 1, parasail.blosum62)
    return out.traceback.query, out.traceback.comp, out.traceback.ref

class RayFlowLightning(L.LightningModule):
    def __init__(self, raygun, esmmodel,
                 rfdenoiser, 
                 no_species,
                 lr = 1e-3, 
                 crossentropyloss = 1., 
                 reconstructionloss = 1., 
                 replicateloss = 1.,
                 log_wandb = False,
                 traininglog = "traininglog.txt",
                 standardlosswt = 1,
                 contrastivewt  = 0.25,
                 denoisewt      = 1, 
                 finetune = False, 
                 function_dim=1000):
        super().__init__()
        self.model            = raygun
        self.esmmodel         = esmmodel
        self.rfdenoiser       = rfdenoiser
        
        assert self.rfdenoiser.sp_st_dim == (1280-function_dim)*50, "Species embeddings mismatch"
        
        self.lr               = lr
        self.crossentropyloss = crossentropyloss
        self.reconstructloss  = reconstructionloss
        self.replicateloss    = replicateloss
        self.trainlosses      = defaultdict(list)
        self.vallosses        = defaultdict(list)
        self.epoch            = 0
        bl                    = substitution_matrices.load("BLOSUM62")
        self.blosummat        = pd.DataFrame(bl, columns = list(bl.alphabet))
        self.blosummat.index  = list(bl.alphabet)
        self.decodermodel     = raygun.esmdecoder
        
        self.esmalphabet      = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 
                                 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
                                 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 
                                 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
        self.toktoalphdict    = {k: i for i, k in self.esmalphabet.items()} 
        
        
        self.embloss          = 0
        
        
        self.log_wandb        = log_wandb
        self.traininglog      = traininglog
        
        # loss regularization
        self.runid            = 0
        self.tlosshistory     = []
        self.coolingtime      = 100
        self.averagingwindow  = 500
        self.std_threshold    = 15
        self.standardlosswt   = standardlosswt
        self.contrastivewt    = contrastivewt
        self.denoisewt        = denoisewt
        self.function_dim     = function_dim
        self.finetune         = finetune
        self.margin           = 0.5
        
    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = [k for k in checkpoint["state_dict"].keys() if "esmmodel" in k]
        for k in keys_to_remove:
            checkpoint["state_dict"].pop(k)

    def log_values(self, batch, losses):
        refb           = batch["reference"]
        refo           = batch["same_tax_different_ortho"]
        reft           = batch["same_ortho_different_tax"]
        outstr         = f"Step {self.global_step}\n"
        for i in range(len(refb["seq"])):
            rnm,  onm,  tnm          = (refb["name"][i].rjust(15), 
                                        refo["name"][i].rjust(15), 
                                        reft["name"][i].rjust(15))
            rseq, oseq, tseq         = (refb["seq"][i], 
                                        refo["seq"][i], 
                                        reft["seq"][i])
            rtax, otax, ttax         = (refb["taxids"][i],
                                        refo["taxids"][i],
                                        reft["taxids"][i])
            
            align_o, bar_o, align_r1 = pairwise(oseq, rseq)
            align_t, bar_t, align_r2 = pairwise(oseq, rseq)
            
            outstr                  += f"""Ref   {rnm} | {rtax} | norm {refb['embed'].norm().item():.5f}
Ortho {onm} | {otax} | norm {refo['embed'].norm().item():.5f}
Taxa  {tnm} | {ttax} | norm {reft['embed'].norm().item():.5f}
-------------
| Alignment |
-------------
Ortho    : {align_o}
         : {bar_o}
         : {align_r1}
Ref
         : {align_r2}
         : {bar_t}
Taxa     : {align_t}
{'-'*100}
"""
        with open(self.traininglog, "a") as logf:
            outstr += "\nLosses: "
            for k, v in losses.items():
                outstr += f"\n{k.rjust(40)} : {float(v):.5f}"
            outstr += "\n" + "#"*100 + "\n"
            logf.write(outstr)
        return

    def configure_optimizers(self):
        if not self.finetune:
            params    = self.model.parameters()
            optimizer = torch.optim.Adam(params, lr = self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler" : {
                "scheduler": scheduler,
                "monitor": "val_blosum_ratio",
                "interval" : "epoch",
                "freq"     : 1
            },
        }
    
    def __get_esm_embedding(self, batchmap):
        tokens, mask       = batchmap["tokens"], batchmap["mask"]
        with torch.no_grad():
            embeddings = self.esmmodel(tokens, repr_layers = [33], 
                                    return_contacts = False)["representations"][33]
            embeddings      = embeddings[:, 1:-1, :] # remove the start token
        tokens              = tokens[:, 1:]
        tokens[tokens == 2] = 1
        tokens              = tokens[:, :-1]
        return embeddings, tokens
    
    def add_esm_embeddings(self, batchmaps):
        for k, batchmap in batchmaps.items():
            embeddings, toks   = self.__get_esm_embedding(batchmap)
            batchmap["embed"]  = embeddings
            batchmap["tokens"] = toks
        return
    
    def _train_each_(self, mask, token, e, key, lossmap=None):
        batch, seq_, _ = e.shape
        payload = self.model(e, mask=mask, token=token)
        res     = payload["reconstructed_embedding"]
        mem     = payload["fixed_length_embedding"]
        closs   = payload["ce_loss"]
        
        if lossmap is not None and isinstance(lossmap, dict):
            lossmap[f"train/fixed-len-norm-{key}"]      = mem.norm().item()
            lossmap[f"train/reconstructed-embed-{key}"] = res.norm().item()
        
        recloss = F.mse_loss(res*mask.unsqueeze(-1), 
                             e*mask.unsqueeze(-1))
        newlens = torch.randint(MINALLOWEDLENGTH, seq_+1, [batch])
        decemb  = self.model.decoder(mem, newlens)
        reploss = F.mse_loss(mem, self.model.encoder(decemb))
        
        for l, v in [("celoss", closs.item()), 
                    ("recloss", recloss.item()),
                    ("reploss", reploss.item())]:
            self.log(f"train/{key}_{l}", v)
            if lossmap is not None and isinstance(lossmap, dict):
                lossmap[f"train/{key}_{l}"] = v
        return closs, recloss, reploss, res, mem
    
    
    def compute_species_function_gain(self, mem, omem, tmem):
        mem_func     =  mem[:, :, :self.function_dim]
        tmem_func    = tmem[:, :, :self.function_dim]
        omem_func    = omem[:, :, :self.function_dim]
        
        mem_species  =  mem[:, :, self.function_dim:]
        tmem_species = tmem[:, :, self.function_dim:]
        omem_species = omem[:, :, self.function_dim:]
        
        species_gain = (mem_species - omem_species).norm() / (mem_species - tmem_species).norm()
        function_gain= (mem_func - tmem_func).norm() / (mem_func - omem_func).norm()
        return species_gain.item(), function_gain.item()
    
    
    def compute_contrastive_loss(self, memdict, taxdict, lossmap=None):
        refmem  = memdict["reference"]
        omem    = memdict["same_ortho_different_tax"]
        tmem    = memdict["same_tax_different_ortho"]
        
        species_gain, function_gain = self.compute_species_function_gain(refmem, omem, tmem)
        refmemf = refmem[:, :, :self.function_dim]
        refmems = refmem[:, :, self.function_dim:]
        omemf   =   omem[:, :, :self.function_dim]
        omems   =   omem[:, :, self.function_dim:]
        tmemf   =   tmem[:, :, :self.function_dim]
        tmems   =   tmem[:, :, self.function_dim:]
        
        lossfunc= F.triplet_margin_loss(refmemf, omemf, tmemf, margin=1, p=2)
        lossspec= F.triplet_margin_loss(refmems, tmems, omems, margin=1, p=2)
        
        # get it from nn embedding
        rspemb  = rearrange(self.rfdenoiser.compute_species_embed(taxdict["reference"]), 
                            "b (n k) -> b n k", n=50)
        ospemb  = rearrange(self.rfdenoiser.compute_species_embed(taxdict["same_ortho_different_tax"]), 
                            "b (n k) -> b n k", n=50)
        tspemb  = rearrange(self.rfdenoiser.compute_species_embed(taxdict["same_tax_different_ortho"]), 
                            "b (n k) -> b n k", n=50)

        eloss   = (F.mse_loss(omems  , ospemb) + 
                   F.mse_loss(tmems  , tspemb) +
                   F.mse_loss(refmems, rspemb))

        self.log(f"train_c/embedding_contrastive_loss", eloss.item())

        eloss   = eloss * self.embloss
        
        if lossmap is not None and isinstance(lossmap, dict):
            lossmap["species-contrastive-loss"]  = lossspec.item()
            lossmap["function-contrastive-loss"] = lossfunc.item()
            lossmap["contrastive-loss"]          = lossspec.item() + lossfunc.item()
        
        self.log(f"train_c/species_contrastive_loss", lossspec.item())
        self.log(f"train_c/function_contrastive_loss", lossfunc.item())
        self.log(f"train_c/contrastive_loss", lossspec.item() + lossfunc.item())
        
        return lossfunc + lossspec + eloss, species_gain, function_gain
    
    
    def compute_denoising_loss(self, memories, taxmap, lossmap):
        start_emb, end_emb = [memories[k] for k in ["reference",
                                                   "same_ortho_different_tax"]]
        start_tax, end_tax = [taxmap[k] for k in ["reference",
                                                 "same_ortho_different_tax"]]
        loss =  self.rfdenoiser.compute_loss(start_emb, start_tax, 
                                             end_emb, end_tax)
        self.log(f"train/denoising_loss", loss.item())
        lossmap["denoising-loss"] = loss.item()
        return loss
    
    
    def training_step(self, batchmap, batch_idx):
        """
        token, embedding and mask should not contain the begin and end tokens
        """
        lossmap                 = {}
        self.add_esm_embeddings(batchmap)
        bm                      = batchmap["reference"]
        bshape, seq_, _         = bm["embed"].shape
        closs, recloss, reploss = 0, 0, 0
        memories                = {}
        ## default Raygun loss
        (closs, recloss, 
         reploss, refres, mem)  = self._train_each_(bm["mask"], 
                                                    bm["tokens"], 
                                                    bm["embed"], 
                                                    "reference", 
                                                    lossmap)
        self.log(f"train/celoss",    closs)
        self.log(f"train/recloss", recloss)
        self.log(f"train/reploss", reploss)
        
        tloss  = (closs   * self.crossentropyloss   + 
                  recloss * self.reconstructloss + 
                  reploss * self.replicateloss)
                 
        ## contrastive loss
        taxmap                               = {k: v["taxonomy"] for k, v in
                                                batchmap.items()}
        memories["reference"]                = mem
        memories["same_ortho_different_tax"] = self.model.encoder(batchmap["same_ortho_different_tax"]["embed"],
                                                                  batchmap["same_ortho_different_tax"]["mask"])
        memories["same_tax_different_ortho"] = self.model.encoder(batchmap["same_tax_different_ortho"]["embed"],
                                                                  batchmap["same_tax_different_ortho"]["mask"])
        contrastiveloss, spgain, funcgain    = self.compute_contrastive_loss(memories,
                                                                             taxmap, 
                                                                             lossmap)
        ## denoising loss
        denoisingloss                        = self.compute_denoising_loss(memories, 
                                                                           taxmap, 
                                                                           lossmap)
        
        
        finalloss = (tloss           * self.standardlosswt + 
                     contrastiveloss * self.contrastivewt +
                     denoisingloss   * self.denoisewt)
        
        
        with torch.no_grad():
            blosumv, blosumr = self.get_blosum_score(refres, 
                                                     batchmap["reference"]["tokens"])
        self.log("train/blosum_score" , blosumv)
        self.log("train/blosum_ratio" , blosumr)
        self.log("train/species_gain" , spgain)
        self.log("train/function_gain", funcgain)
        
        self.tlosshistory = self.tlosshistory[-self.averagingwindow:]
        
        self.log_values(batchmap, lossmap)
        
        self.runid       += 1
        if self.runid < self.coolingtime:
            self.tlosshistory.append(finalloss.item())
            return finalloss
        
        running_avg = np.mean(self.tlosshistory)
        running_std = np.std(self.tlosshistory)
        
        if tloss.item() >= running_avg + self.std_threshold * running_std:
            tloss_ = float(finalloss.item())
            return finalloss / tloss_ * 1e-9 ## this would ignore the batch
        else:
            self.tlosshistory.append(finalloss.item())
        return finalloss
    

    def validation_step(self, batchmap, batch_idx):
        self.add_esm_embeddings(batchmap)
        payload      = self.model(batchmap["reference"]["embed"], 
                                     mask=batchmap["reference"]["mask"])
        result       = payload["reconstructed_embedding"]
        mem          = payload["fixed_length_embedding"]
        
        
        blosum_curr, blosum_curr_ratio = self.get_blosum_score(result,
                                                                batchmap["reference"]["tokens"])
        
        # check the contrastive loss too
        tmem         = self.model.encoder(batchmap["same_tax_different_ortho"]["embed"])
        omem         = self.model.encoder(batchmap["same_ortho_different_tax"]["embed"])
        
        species_gain, func_gain = self.compute_species_function_gain(mem, omem, tmem)
        
        self.log("val/function_gain",func_gain)
        self.log("val/species_gain", species_gain)
        self.log("val/blosum_score", blosum_curr)
        self.log("val/blosum_ratio", blosum_curr_ratio)
        self.log("val_blosum_score", blosum_curr)
        self.log("val_blosum_ratio", blosum_curr_ratio)
 
    def convert_tokens_to_alph(self, token, lengths):
        """
        token: tensor [batch, seqlen]
        """
        assert len(token.shape) == 2
        batch, _ = token.shape
        alphabets = []
        for i in range(batch):
            li  = lengths[i]
            tok = token[i][:li].tolist() 
            alphabets.append([self.toktoalphdict[t] for t in tok])
        return alphabets
    
    def return_sequences_from_embs(self, embeddings, lengths = None):
        """
        embedding = [batch, seq, dim]
        """
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
        b, n, d = embeddings.shape
        if b != 1:
            assert lengths is not None and len(lengths) == b, \
            "for larger batches, you need to specify the lengths. Additionally, the #lengths should equal the batch size"
        else:
            lengths = [n]
        pred_alphs = []
        for i in range(b):
            logits = self.model.esmdecoder(embeddings[i][None, :lengths[i], :])
            pred_token = torch.argmax(logits, dim = -1).cpu().numpy()
            pred_alph  = self.convert_tokens_to_alph(pred_token, [lengths[i]])
            pred_alph  = "".join(pred_alph[0])
            if b == 1:
                return pred_alph
            else:
                pred_alphs.append(pred_alph)
        return pred_alphs
        
    def get_blosum_score(self, embedding, true_token):
        """
        embedding: tensor [batch, seqlen, dim]
        true_token: tensor [batch, seqlen]
        """
        ## logging.info(f"Tokens shape {true_token.shape}, embed shape {embedding.shape}")
        batch, _, _ = embedding.shape
        lengths     = []
        
        for i in range(batch):
            tok  = true_token[i]
            lengths.append(tok[tok != 1].shape[0]) # tok being 1 implies padding
        with torch.no_grad():
            true_alph    = self.convert_tokens_to_alph(true_token.cpu().numpy(),
                                                       lengths)
            logits       = self.model.esmdecoder(embedding)
            pred_tokens  = torch.argmax(logits, dim = -1).cpu().numpy()
            pred_alph    = self.convert_tokens_to_alph(pred_tokens, lengths)
            blcs, blrs   = [], []
            for b in range(batch):
                blc, blr       = self.compute_blosum_score(true_alph[b], 
                                                           pred_alph[b])
                blcs.append(blc)
                blrs.append(blr)
        return np.average(blcs), np.average(blrs)

    def compute_blosum_score(self, true, predicted):
        blosum_max  = 0
        blosum_curr = 0
        for p, q in zip(true, predicted):
            try:
                blosum_c_score = self.blosummat.loc[p.upper(), 
                                                    q.upper()] # if no p and q, this triggers exception
            except Exception as e:
                blosum_c_score = 0
            try:
                blosum_max += self.blosummat.loc[p.upper(), 
                                                 p.upper()]
            except Exception as e:
                blosum_max += 1
            blosum_curr += blosum_c_score
        return blosum_curr, blosum_curr / blosum_max