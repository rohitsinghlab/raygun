# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import numpy as np
import pandas as pd
from tqdm import tqdm 
from glob import glob
from Bio import SeqIO
import biotite.structure.io as bsio
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
import torch
import os
from subprocess import Popen, PIPE
import esm
from Bio import pairwise2 as pw
from Bio.Align import substitution_matrices

def getplddt(pdbfile):
    cmd = ["awk", "BEGIN{plddt=0;count=0} {plddt+=$11;count+=1;} END{print plddt / count;}", pdbfile]
    process = Popen(cmd, stdout = PIPE, stderr = PIPE)
    out, err = process.communicate()
    return float(out.decode())

def getperplexity(pdbfile, model, alphabet):
    struct = esm.inverse_folding.util.load_structure(pdbfile, "A")
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(struct)
    ll_fullseq, ll_withcoord = esm.inverse_folding.util.score_sequence(model, alphabet, 
                                                                       coords, native_seq)
    return np.exp(-ll_fullseq), len(native_seq)

def gettmscore(genpdb, refpdb):
    rstruct = get_structure(refpdb)
    gstruct = get_structure(genpdb)
    # get residue data
    rcoords, rseq = get_residue_data(next(rstruct.get_chains()))
    gcoords, gseq = get_residue_data(next(gstruct.get_chains()))

    res = tm_align(rcoords, gcoords, rseq, gseq)
    score = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2
    return score

def getsequenceidentity(gen, ref, blosum62, ispdb = True, method = "default"):
    if ispdb:
        genstruct = esm.inverse_folding.util.load_structure(gen, "A")
        refstruct = esm.inverse_folding.util.load_structure(ref, "A")
        _, genseq = esm.inverse_folding.util.extract_coords_from_structure(genstruct)
        _, refseq = esm.inverse_folding.util.extract_coords_from_structure(refstruct)
    else:
        refseq    = ref
        genseq    = gen
    align = pw.align.globaldx(refseq,
                              genseq, 
                              blosum62)[0]
    aligned_seq1, aligned_seq2 = align[0], align[1]
    if method == "ignore-dash":
        zipped = [(r1, r2) for r1, r2 in zip(aligned_seq1, aligned_seq2) if (r1 != "-" and r2 != "-")]
        aligned_seq1 , aligned_seq2 = zip(*zipped)
        aligned_seq1 = "".split(aligned_seq1)
        aligned_seq2 = "".split(aligned_seq2)

    matches = sum(res1 == res2 for res1, res2 in zip(aligned_seq1, aligned_seq2))
    alignment_length = len(aligned_seq1)
    sequence_identity = matches / alignment_length
    return sequence_identity

