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

MINALLOWEDLENGTH = 50

class RaygunLightning(L.LightningModule):
    def __init__(self, raygun, lr = 1e-3, 
                crossentropyloss = 1., 
                reconstructionloss = 1., 
                replicateloss = 1.,
                log_wandb = False,
                traininglog = "traininglog.txt"):
        super().__init__()
        self.model  = raygun
        self.lr     = lr
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
        
        self.log_wandb        = log_wandb
        self.traininglog      = traininglog
        
        # loss regularization
        self.runid            = 0
        self.tlosshistory     = []
        self.coolingtime      = 100
        self.averagingwindow  = 500
        self.std_threshold    = 15

    def log_error(self, batch, loss):
        idx, seq       = zip(*batch)
        df             = pd.DataFrame({"id" : idx, 
                                       "seq" : seq})
        running_avg    = np.mean(self.tlosshistory)
        running_std    = np.std(self.tlosshistory)
        with open(self.traininglog, "a") as logf:
            logf.write(f"\n\nEpoch {self.epoch}, run {self.runid}")
            logf.write(f"\n\tBatch Loss   : {loss}\n")
            logf.write(f"\n\tRunning Loss : {running_avg}" 
                       + u" \u00B1 " 
                       + f"{running_std}\n")
            logf.write(df.to_string())
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
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

    def training_step(self, batch, batch_idx):
        """
        token, embedding and mask should not contain the begin and end tokens
        """
        tokens, e, mask, binfo = batch
        bshape, seq_, _        = e.shape
        if mask is None:
            assert bshape == 1, "Batch is larger than 1 but no mask provided"
            ## required when replicateloss > 0
            newlengths = torch.randint(MINALLOWEDLENGTH, seq_, [1])
        else:
            lengths    = mask.sum(dim = 1)
            newlengths = torch.concat([torch.randint(MINALLOWEDLENGTH, l, [1]) 
                         for l in lengths]) 
        tloss = 0
        if self.crossentropyloss > 0:
            payload    = self.model(e, mask = mask, token = tokens)
            result     = payload["reconstructed_embedding"]
            mem        = payload["fixed_length_embedding"]
            crossloss  = payload["ce_loss"]
            tloss      = tloss + self.crossentropyloss * crossloss
            self.trainlosses["Cross-Entropy Loss"].append(crossloss.item())
            self.log("Cross-Entropy Loss", crossloss.item() if crossloss.item() < 10 else 10)
        else:
            payload    = self.model(e, mask = mask)
            result     = payload["reconstructed_embedding"]
            mem        = payload["fixed_length_embedding"]
            
        if self.reconstructloss > 0:
            recloss    = F.mse_loss(result * mask.unsqueeze(-1), e * mask.unsqueeze(-1))
            tloss      = tloss + self.reconstructloss * recloss
            self.trainlosses["Reconstruction Loss"].append(recloss.item())
            self.log("Reconstruction Loss", recloss.item() if recloss.item() < 10 else 10)
        if self.replicateloss > 0:
            decodedemb = self.model.decoder(mem, newlengths)
            reploss    = F.mse_loss(mem, self.model.encoder(decodedemb)) 
            tloss      = tloss + self.replicateloss * reploss 
            self.trainlosses["Replicate Loss"].append(reploss.item())
            self.log("Replicate Loss", reploss.item() if reploss.item() < 10 else 10)
        blosumv, blosumr = self.get_blosum_score(result.detach(), tokens.detach())
        self.log("Blosum score", blosumv)
        self.log("Blosum ratio", blosumr)
        
        self.tlosshistory = self.tlosshistory[-self.averagingwindow:]
        
        self.runid       += 1
        if self.runid < self.coolingtime:
            self.tlosshistory.append(tloss.item())
            return tloss
        
        running_avg = np.mean(self.tlosshistory)
        running_std = np.std(self.tlosshistory)
        
        if tloss.item() >= running_avg + self.std_threshold * running_std:
            self.log_error(binfo, tloss.item())
            tloss_ = float(tloss.item()) * 0.01
            return tloss / tloss_ * running_avg ## this would ignore the batch
        else:
            self.tlosshistory.append(tloss.item())
            return tloss

    def on_train_epoch_end(self):
        logf = f"Completed Training Epoch {self.epoch+1}: "
        for k, v in self.trainlosses.items():
            logf += f"{k} : {np.mean(v):.4f}"
        logging.info(logf)
        self.trainlosses = defaultdict(list)
        self.epoch      += 1
        return

    def validation_step(self, batch, batch_idx):
        tokens, e, mask, _ = batch
        payload            = self.model(e, mask = mask)
        result             = payload["reconstructed_embedding"]
        mem                = payload["fixed_length_embedding"]
        blosum_curr, blosum_curr_ratio = self.get_blosum_score(result,
                                                                tokens)
        self.log("val_blosum_score", blosum_curr)
        self.log("val_blosum_ratio", blosum_curr_ratio)
        self.vallosses["Blosum Score"].append(blosum_curr)
        self.vallosses["Blosum ratio"].append(blosum_curr_ratio)

    def on_validation_epoch_end(self):
        logf = f"Completed Validation Epoch {self.epoch}"
        for k, v in self.vallosses.items():
            logf += f"{k} : {np.mean(v): .4f}"
        self.validlosses = defaultdict(list)
        return

    ### Blosum scores prediction 
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
                blosum_max += self.blosummat.loc[p.upper(), 
                                                 p.upper()]
                blosum_curr += blosum_c_score
            except Exception as e:
                continue
        return blosum_curr, blosum_curr / blosum_max