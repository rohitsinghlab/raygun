from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinTensor
from esm.tokenization import get_esmc_model_tokenizers
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

def get_logits_masked_marginals(model, tokenizers, seq, device=0):
    model.eval()
    log_probs  = []
    for i in range(len(seq)):
        aa                = seq[i]
        aa_idx            = tokenizers.get_vocab()[aa]
        seq_masked        = model.encode(ESMProtein(sequence=seq[:i] + "<mask>" + seq[i+1:]))
        with torch.no_grad():
            logits_output = model.logits(seq_masked, LogitsConfig(sequence=True, return_embeddings=False))
            logits_tensor = logits_output.logits.sequence # shape: [1, seq_len, vocab_size]
        log_probs.append(torch.log_softmax(logits_tensor, dim=-1)[0, i, aa_idx].item())
    
    probs = torch.exp(torch.tensor(log_probs))
    probs = probs / torch.sum(probs)
    return probs

def main(args):
    os.makedirs(args.output_folder, 
                exist_ok=True)
    device          = args.device
    seq             = str(SeqIO.read(args.fasta, "fasta").seq)
    print(f"Seq to use for generation: {seq}")

    model = ESMC.from_pretrained("esmc_300m").to(device)
    tokenizers = get_esmc_model_tokenizers()

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
            tprobs      = get_logits_masked_marginals(model, tokenizers, seqt,
                                                      device=0)
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