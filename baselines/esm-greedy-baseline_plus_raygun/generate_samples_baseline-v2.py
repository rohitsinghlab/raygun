from esm.pretrained import esm2_t33_650M_UR50D
from esm import MSATransformer
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", help="Fasta file with one seq")
    parser.add_argument("--output-folder", help="Output folder")
    parser.add_argument("--no-to-gen", default=10, type=int, help="No of seq to generate")
    parser.add_argument("--min-length", type=int, help="Minimum length")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--max-length", type=int, help="Maximum length")
    parser.add_argument("--prefix", help="prefix to be added to each record")
    args = parser.parse_args()
    return args

import os
import random

def get_logits_masked_marginals(model, alphabet, seq, device=0):
    assert next(iter(model.parameters())).device.index==device, "Device mismatch"
    data = [("protein", seq)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    log_probs = []
    for i in range(len(seq)):
        batch_tokens_masked         = batch_tokens.clone()
        batch_tokens_masked[0, i+1] = alphabet.mask_idx #+1 to ignore the start token
        prob_idx                    = alphabet.get_idx(seq[i])
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
        log_probs.append(token_probs[0, i+1, prob_idx].item())
    probs = torch.exp(torch.tensor(log_probs))
    probs = probs / torch.sum(probs)
    return probs

def main(args):
    os.makedirs(args.output_folder, 
                exist_ok=True)
    device          = args.device
    seq             = str(SeqIO.read(args.fasta, "fasta").seq)
    print(f"Seq to use for generation: {seq}")

    model, alphabet = esm2_t33_650M_UR50D()
    model           = model.to(device)
    model.eval()
    records         = []
    
    for i in tqdm(range(args.no_to_gen), total=args.no_to_gen, 
                 desc="Generating sequences"):
        targetlen       = np.random.randint(args.min_length, 
                                 args.max_length, 
                                 [1])[0]
        seqlen          = len(seq)
        seqt            = seq
        for j in range(seqlen-targetlen):
            tprobs      = get_logits_masked_marginals(model, alphabet, seqt,
                                                      device=device)
            to_choose   = sorted(np.random.choice(len(seqt), size=len(seqt)-1,
                                            replace=False, 
                                            p=tprobs.numpy()))
            seqt_       ="".join(seqt[t] for t in to_choose)
            seqt        = seqt_
        idx  = f"{args.prefix}-len-{targetlen}-idx-{i}"
        rec = SeqRecord(id=idx, 
                       seq=Seq(seqt),
                       description="")
        records.append(rec)
        
    out_file=f"{args.output_folder}/{args.prefix}-{args.min_length}-{args.max_length}_{args.no_to_gen}-full.fasta"
    SeqIO.write(records, out_file, "fasta")
    
if __name__ == "__main__":
    args = get_args()
    main(args)