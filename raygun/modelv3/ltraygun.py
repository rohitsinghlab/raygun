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

MINALLOWEDLENGTH = 10

class RaygunLightning(L.LightningModule):
    def __init__(self, raygun, emodel,
                lr = 1e-3, 
                crossentropyloss = 1., 
                reconstructionloss = 1., 
                replicateloss = 1.,
                log_wandb = False,
                traininglog = "traininglog.txt",
                finetune = False,
                esmalphabet = None,
                e1 = False):
        super().__init__()
        self.model            = raygun
        self.emodel           = emodel
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
        self.decodermodel     = raygun.e1decoder


        print(e1, " Using e1 T/F")
        self.e1 = e1

        if e1:
            # extract from emodel if possible
            self.esmalphabet  = getattr(self.emodel, "token_to_id", None) or {}
        else:
            # ESM defaults
            self.esmalphabet  = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 
                                'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
                                'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 
                                'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

        self.toktoalphdict    = {v: k for k, v in self.esmalphabet.items()}

        self.pad_id           = self.esmalphabet.get("<pad>") if self.esmalphabet else None
        self.mask_id          = self.esmalphabet.get("<mask>") if self.esmalphabet else None
        self.bos_id           = self.esmalphabet.get("<bos>") or self.esmalphabet.get("<cls>") or None
        self.eos_id           = self.esmalphabet.get("<eos>") if self.esmalphabet else None

        self.get_embeddings   = self.get_e1_embeddings if e1 else self.get_esm_embeddings
        
        self.log_wandb        = log_wandb
        self.traininglog      = traininglog
        
        # loss regularization
        self.runid            = 0
        self.tlosshistory     = []
        self.coolingtime      = 100
        self.averagingwindow  = 500
        self.std_threshold    = 15
        self.finetune         = finetune



    def on_save_checkpoint(self, checkpoint):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        keys_to_remove = [k for k in checkpoint["state_dict"].keys() if "emodel" in k]
        for k in keys_to_remove:
            checkpoint["state_dict"].pop(k)

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

    # def configure_optimizers(self):
    #     if not self.finetune:
    #         optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
    #     else:
    #         optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr = self.lr)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='max',
    #         factor=0.5,
    #         patience=5,
    #         min_lr=1e-6,
    #     )
    #     # Return optimizer and scheduler
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler" : {
    #             "scheduler": scheduler,
    #             "monitor": "val_blosum_ratio",
    #             "interval" : "epoch",
    #             "freq"     : 1
    #         },
    #     }

    def configure_optimizers(self):
        if not self.finetune:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr)
        return optimizer

    def get_esm_embeddings(self, batch):
        tokens, mask, binfo    = batch
        with torch.no_grad():
            embeddings = self.emodel(tokens, repr_layers = [33], 
                                    return_contacts = False)["representations"][33]
            embeddings      = embeddings[:, 1:-1, :] # remove the start token
        tokens              = tokens[:, 1:]
        tokens[tokens == 2] = 1
        tokens              = tokens[:, :-1]
        return tokens, embeddings, mask, binfo

    def get_e1_embeddings(self, batch):
        # TODO: Cleanup batch handling
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", (input_ids != self.pad_id).long() if self.pad_id is not None else torch.ones_like(input_ids))
            binfo = batch.get("raw", None)
        else:
            # legacy: if dataset still yields tuple, preserve compatibility
            try:
                tokens, mask, binfo = batch
                input_ids = tokens
                attention_mask = mask
            except Exception:
                raise ValueError("Unknown batch format in get_esm_embeddings")

        # device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else input_ids.device
        device = self.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            embeddings = self.emodel(input_ids=input_ids, attention_mask=attention_mask)

        # no shifting for e1
        tokens = input_ids

        if attention_mask is None and self.pad_id is not None:
            attention_mask = (tokens != self.pad_id).long()

        return tokens, embeddings, attention_mask, binfo

            
    
    def training_step(self, batch, batch_idx):
        """
        token, embedding and mask should not contain the begin and end tokens
        This version includes debug/mapping to prevent CUDA asserts from CE loss.
        """
        # --- existing beginning: get embeddings from esmmodel/adapter ---
        tokens, e, mask, binfo = self.get_embeddings(batch)
        bshape, seq_, _        = e.shape

        # --- MAP E1 token IDs -> decoder alphabet IDs (cached map) ---
        if self.e1:
            if not hasattr(self, "_e1_to_alpha_map"):
                # Build id_to_token mapping from adapter
                id_to_token = getattr(self.emodel, "id_to_token", None)
                if id_to_token is None:
                    token_to_id = getattr(self.emodel, "token_to_id", {})
                    id_to_token = {v: k for k, v in token_to_id.items()}

                if len(id_to_token) == 0:
                    # fallback: no adapter map available (shouldn't happen)
                    raise RuntimeError("E1 adapter has no id->token mapping (id_to_token). Cannot map labels.")

                e1_vocab_size = max(id_to_token.keys()) + 1
                pad_fill = self.pad_id if getattr(self, "pad_id", None) is not None else 0

                mapper = torch.full((e1_vocab_size,), fill_value=pad_fill, dtype=torch.long, device='cpu')

                # self.esmalphabet should be token_str -> alpha_id (old alphabet mapping)
                tokenstr_to_alphaid = self.esmalphabet if isinstance(self.esmalphabet, dict) else {}
                for e1_id, tokstr in id_to_token.items():
                    mapper[e1_id] = tokenstr_to_alphaid.get(tokstr, pad_fill)

                # cache mapper on module (CPU)
                self._e1_to_alpha_map = mapper

            # apply mapping to tokens (move mapper to same device)
            mapper = self._e1_to_alpha_map.to(tokens.device)
            tokens_mapped = mapper[tokens]    # same shape as tokens

            # # Optional debug print to see some mapped values (first few)
            # if batch_idx == 0:
            #     sample_map_preview = mapper[:min(len(mapper), 20)].tolist()
            #     print("DEBUG: e1->alpha mapper preview (first 20):", sample_map_preview)
            #     print("DEBUG: tokens (first row) sample E1 ids:", tokens[0, :10].detach().cpu().tolist())
            #     print("DEBUG: tokens_mapped (first row) sample alpha ids:", tokens_mapped[0, :10].detach().cpu().tolist())

            # Use tokens_mapped as labels for downstream calls
            tokens_for_loss = tokens_mapped

        # --- Now proceed with normal training logic, but make CE loss robust ---
        bshape, seq_, _ = e.shape
        if mask is None:
            assert bshape == 1, "Batch is larger than 1 but no mask provided"
            newlengths = torch.randint(MINALLOWEDLENGTH, seq_+1, [1])
        else:
            lengths    = mask.sum(dim = 1)
            newlengths = torch.concat([torch.randint(MINALLOWEDLENGTH, l+1, [1]) for l in lengths])

        tloss = 0

        if self.crossentropyloss > 0:
            # call model with mapped token ids
            payload    = self.model(e, mask = mask, token=tokens_for_loss if self.e1 else tokens)
            result     = payload["reconstructed_embedding"]
            mem        = payload["fixed_length_embedding"]
            crossloss  = payload["ce_loss"]

            tloss      = tloss + self.crossentropyloss * crossloss

            self.trainlosses["Cross-Entropy Loss"].append(crossloss.item())
            self.log("Cross-Entropy Loss", crossloss.item() if crossloss.item() < 10 else 10)

            # # safe append/ log inside try to avoid masked CUDA asserts
            # try:
            #     cl_val = float(crossloss.detach().cpu().item())
            # except Exception:
            #     cl_val = None
            # self.trainlosses["Cross-Entropy Loss"].append(cl_val if cl_val is not None else 0.0)
            # if cl_val is not None:
            #     self.log("Cross-Entropy Loss", cl_val if cl_val < 10 else 10)
        else:
            payload    = self.model(e, mask = mask)
            result     = payload["reconstructed_embedding"]
            mem        = payload["fixed_length_embedding"]

        # ... rest of your losses and logging unchanged ...
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
        
        if self.e1:
            blosumv, blosumr = self.get_blosum_score(result.detach(), tokens_for_loss.detach())
        else:
            blosumv, blosumr = self.get_blosum_score(result.detach(), tokens.detach())
        self.log("Blosum score", blosumv)
        self.log("Blosum ratio", blosumr)
        self.log("global_step", float(self.global_step))

        # remaining bookkeeping
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
            return tloss / tloss_ * running_avg
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
        tokens, e, mask, _ = self.get_embeddings(batch)
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
            logits = self.model.e1decoder(embeddings[i][None, :lengths[i], :])
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
        # logging.info(f"Tokens shape {true_token.shape}, embed shape {embedding.shape}")
        batch, _, _ = embedding.shape
        lengths     = []
        
        for i in range(batch):
            tok  = true_token[i]
            lengths.append(tok[tok != self.pad_id].shape[0])
        with torch.no_grad():
            true_alph    = self.convert_tokens_to_alph(true_token.cpu().numpy(),
                                                       lengths)
            logits       = self.model.e1decoder(embedding)
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
                                                    q.upper()] 
                # if no p and q, this triggers exception
                blosum_max += self.blosummat.loc[p.upper(), 
                                                 p.upper()]
                blosum_curr += blosum_c_score
            except Exception as e:
                continue
        if blosum_max == 0:
            return blosum_curr, 0.0
        return blosum_curr, blosum_curr / blosum_max

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning-native prediction step.
        Identical precision + vocab handling to training_step.
        """
        


        tokens, e, mask, binfo = self.get_embeddings(batch)

        # Same forward path as training / validation
        payload = self.model(
            e,
            mask=mask,
        )

        result             = payload["reconstructed_embedding"]

        blosum_curr, blosum_curr_ratio = self.get_blosum_score(result,
                                                                tokens)

        print(f"Batch {batch_idx} - Blosum Score: {blosum_curr}, Blosum Ratio: {blosum_curr_ratio}")

        logits = payload["logits"] if "logits" in payload else self.model.e1decoder(
            payload["reconstructed_embedding"]
        )

        pred_tokens = torch.argmax(logits, dim=-1)

        # -----------------------------
        # DEBUG: raw token comparison
        # -----------------------------
        if batch_idx == 0:
            id_to_token = getattr(self.emodel, "id_to_token", None)
            if id_to_token is None:
                tok_to_id = getattr(self.emodel, "token_to_id", {})
                id_to_token = {v: k for k, v in tok_to_id.items()}

            print("\n=== DEBUG RAW TOKENS ===")
            print("TRUE token ids:", tokens[0, :-1].detach().cpu().tolist())
            print("TRUE token strs:", [id_to_token.get(t, "?") for t in tokens[0, :-1].cpu().tolist()])
            print("PRED token ids:", pred_tokens[0, :-1].detach().cpu().tolist())
            print("PRED token strs:", [id_to_token.get(t, "?") for t in pred_tokens[0, :-1].cpu().tolist()])
            print("========================\n")
        
        print(payload["reconstructed_embedding"].shape, tokens.shape)
        
        score = self.get_blosum_score(payload["reconstructed_embedding"], tokens)

        # -----------------------------
        # 🔑 E1 → AA alphabet mapping (CRITICAL)
        # -----------------------------
        if self.e1:
            if not hasattr(self, "_e1_to_alpha_map"):
                id_to_token = getattr(self.emodel, "id_to_token", None)
                if id_to_token is None:
                    tok_to_id = getattr(self.emodel, "token_to_id", {})
                    id_to_token = {v: k for k, v in tok_to_id.items()}

                e1_vocab_size = max(id_to_token.keys()) + 1
                pad_fill = self.pad_id if self.pad_id is not None else 0

                mapper = torch.full(
                    (e1_vocab_size,),
                    fill_value=pad_fill,
                    dtype=torch.long,
                )

                # token_str → AA-id mapping
                for e1_id, tokstr in id_to_token.items():
                    mapper[e1_id] = self.esmalphabet.get(tokstr, pad_fill)

                self._e1_to_alpha_map = mapper

            pred_tokens = self._e1_to_alpha_map.to(pred_tokens.device)[pred_tokens]

        # -----------------------------
        # Convert to amino-acid strings
        # -----------------------------
        lengths = (
            mask.sum(dim=1).tolist()
            if mask is not None
            else [pred_tokens.shape[1]] * pred_tokens.shape[0]
        )

        pred_tokens_np = pred_tokens.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None

        pred_alph = []
        for i in range(pred_tokens_np.shape[0]):
            if mask_np is not None:
                valid_idx = mask_np[i].astype(bool)
                toks = pred_tokens_np[i][valid_idx]
            else:
                toks = pred_tokens_np[i][:lengths[i]]

            pred_alph.append(
                [self.toktoalphdict[t] for t in toks]
            )


        outputs = []
        for i in range(len(pred_alph)):
            out = {
                "pred": "".join(pred_alph[i]),
                "score": score,
            }
            if binfo is not None:
                idx, true_seq = binfo[i]
                out["id"] = idx
                out["true"] = true_seq
            outputs.append(out)

        return outputs
