import lmdb
import pickle as pkl
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
import torch
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
import random

def put_to_lmdb(txn, key, val):
    txn.put(pkl.dumps(key), pkl.dumps(val))
    return

def get_from_lmdb(txn, key):
    val = pkl.loads(txn.get(pkl.dumps(key)))
    return val

def save_lmdb(OUTLMDB, trajtrain):
    env       = lmdb.open(OUTLMDB, map_size=2*104857600)  # 1 GB max size
    taxcounts = defaultdict(int)
    protcount = 0

    with env.begin(write=True) as txn:
        for oid, (k, df) in enumerate(trajtrain.items()):
            put_to_lmdb(txn, ("orthoid?", oid), k)                           # given some idx, return the corresponding ortho-id
            for idx, (prot, seq, taxid) in enumerate(df[["prot", "seq", 
                                        "taxonomy-class"]].values):
                tax_id_curr = taxcounts[taxid]
                put_to_lmdb(txn, ("prots?", protcount), prot)                # given some idx, return the corresponding protein
                protcount  += 1

                # save protein info
                put_to_lmdb(txn, ("sequence?", prot), seq)                   # query a protein name for sequence  
                put_to_lmdb(txn, ("taxonomy?", prot), taxid)                 # query a protein name for its taxid
                put_to_lmdb(txn, ("orthoid?", prot), k)                      # query a protein for its ortholog-id

                # save ortholog info
                put_to_lmdb(txn, ("prots_in_orthoid?", k, idx), prot)        # query a protein at a particular id, given the ortholog-id
                put_to_lmdb(txn, ("prots_in_taxonomy?", taxid, tax_id_curr), 
                           prot)                                             # query a protein at a particular id, given the ortholog-id

                taxcounts[taxid] += 1
            put_to_lmdb(txn, ("#prots_in_orthoid?", k), len(df))             # query the total number of prots in an orthoid

        for idx, (tax, count) in enumerate(taxcounts.items()):               # number of proteins a taxonomy class has
            put_to_lmdb(txn, ("taxonomy?", idx), tax)
            put_to_lmdb(txn, ("#prots_in_taxonomy?", tax), count)

        put_to_lmdb(txn, "#taxonomy?", len(taxcounts))
        put_to_lmdb(txn, "#prots?", protcount)
        put_to_lmdb(txn, "#orthoid?", len(trajtrain))
    return

class RaygunContrastiveDataset(Dataset):
    def __init__(self, lmdb_file, esmalphabet,
                tries=10,
                nearestnghbrs=10):
        self.env = lmdb.open(lmdb_file, lock=False, readonly=True)
        with self.env.begin() as txn:
            self.no_prots = self.get_from_lmdb(txn, "#prots?")
            self.no_ortho = self.get_from_lmdb(txn, "#orthoid?")
            self.no_tax   = self.get_from_lmdb(txn, "#taxonomy?")
            self.taxmap   = self.tokenize_taxonomy(txn)
        self.bc           = esmalphabet.get_batch_converter()
        self.tries        = tries
        self.nearestnghbrs= nearestnghbrs
        return 
    
    def tokenize_taxonomy(self, txn):
        taxmap = {}
        for i in range(self.no_tax):
            tax         = get_from_lmdb(txn, 
                                        ("taxonomy?", i))
            taxmap[tax] = i
        return taxmap
    
    def get_from_lmdb(self, txn, key):
        val = pkl.loads(txn.get(pkl.dumps(key)))
        return val
    
    def __len__(self): 
        return self.no_prots
    
    def get_prot_features(self, txn, prot):
        seq   = self.get_from_lmdb(txn, ("sequence?", prot))
        tax   = self.get_from_lmdb(txn, ("taxonomy?", prot))
        ortho = self.get_from_lmdb(txn, ("orthoid?", prot))
        return seq, tax, ortho
    
    def return_sodt_stdo(self, txn, prot, topk=10): 
        # taxonomy and immediate neighbors
        tax             = get_from_lmdb(txn, ("taxonomy?", prot))
        tax_ranking     = get_from_lmdb(txn, ("tax_ranking?", tax))
        tax_ranking_map = {k: i for i, k in 
                           enumerate(tax_ranking)}
        # random orthologs
        orthos  = get_from_lmdb(txn, ("orthoids?", prot))
        oid     = random.sample(orthos, 1)[0]
        nprots  = get_from_lmdb(txn, ("#prots_in_orthoid?", oid))
        pid     = int(np.random.randint(0, nprots, [1])[0])
        oprot   = get_from_lmdb(txn, ("prots_in_orthoid?", oid, pid))

        # get taxonomy of the neighboring prot
        # if taxonomy belongs to immediate neighbors, described by `k`, return None
        otax    = get_from_lmdb(txn, ("taxonomy?", oprot))
        rank    = tax_ranking_map[otax]

        if rank < topk:
            return None

        # if taxonomy is not in the immediate neighbor
        # get the neighbor rank
        sim_taxid = int(np.random.randint(0, min(50, rank - (topk//2)),
                                          [1])[0])
        
        sim_tax   = tax_ranking[sim_taxid]
        # select a protein belong to the sim tax
        tprot_id  = int(np.random.randint(0, get_from_lmdb(txn, 
                                                        ("#prots_in_taxonomy?", 
                                                         sim_tax)), 
                                          [1])[0])
        tprot     = get_from_lmdb(txn, ("prots_in_taxonomy?", sim_tax, 
                                        tprot_id))
        # check if tprot has the same ortholog id
        tprot_orth= get_from_lmdb(txn, ("orthoids?", tprot))
        if len(set(orthos).intersection(tprot_orth)) != 0:
            return None

        oseq      = get_from_lmdb(txn, ("sequence?", oprot))
        tseq      = get_from_lmdb(txn, ("sequence?", tprot))
        rseq      = get_from_lmdb(txn, ("sequence?", prot))

        oorthos   = get_from_lmdb(txn, ("orthoids?", oprot))

        return {"same_ortho_diff_tax": {"prot": oprot, "seq": oseq, 
                                        "ortho": oorthos, "tax": otax}, 
                "same_tax_diff_ortho": {"prot": tprot, "seq": tseq, 
                                        "tax": sim_tax, "ortho": tprot_orth}, 
                "ref"                : {"prot": prot,  "seq": rseq, 
                                        "tax": tax, "ortho": orthos}}
    
    
    def return_sodt_stdo_(self, txn, prot, topk=10):
        # taxonomy and immediate neighbors
        tax              = get_from_lmdb(txn, ("taxonomy?", prot))
        all_nghbrs       = get_from_lmdb(txn, ("neighbor_of_tax?", tax))
        immediate_nghbrs = set(all_nghbrs[:topk])
        nghbrs_rank      = {k: i for i, k in 
                            enumerate(all_nghbrs)}
        # random orthologs
        orthos  = get_from_lmdb(txn, ("orthoids?", prot))
        oid     = random.sample(orthos, 1)[0]
        nprots  = get_from_lmdb(txn, ("#prots_in_orthoid?", oid))
        pid     = int(np.random.randint(0, nprots, [1])[0])
        oprot   = get_from_lmdb(txn, ("prots_in_orthoid?", oid, pid))

        # get taxonomy of the neighboring prot
        # if taxonomy belongs to immediate neighbors, return None
        otax    = get_from_lmdb(txn, ("taxonomy?", oprot))
        if otax in immediate_nghbrs:
            return None

        # if taxonomy is not in the immediate neighbor
        # get the neighbor rank
        rank    = None if otax not in nghbrs_rank \
                  else nghbrs_rank[otax]
        if rank is not None:
            sim_taxid = int(np.random.randint(0, int(rank / 1.5),
                                              [1])[0])
        else:
            sim_taxid = int(np.random.randint(0, len(all_nghbrs), 
                                              [1])[0])
        sim_tax   = all_nghbrs[sim_taxid]
        # select a protein belong to the sim tax
        tprot_id  = int(np.random.randint(0, get_from_lmdb(txn, 
                                                        ("#prots_in_taxonomy?", 
                                                         sim_tax)), 
                                          [1])[0])
        tprot     = get_from_lmdb(txn, ("prots_in_taxonomy?", sim_tax, 
                                        tprot_id))
        # check if tprot has the same ortholog id
        tprot_orth= get_from_lmdb(txn, ("orthoids?", tprot))
        if len(set(orthos).intersection(tprot_orth)) != 0:
            return None

        oseq      = get_from_lmdb(txn, ("sequence?", oprot))
        tseq      = get_from_lmdb(txn, ("sequence?", tprot))
        rseq      = get_from_lmdb(txn, ("sequence?", prot))

        oorthos   = get_from_lmdb(txn, ("orthoids?", oprot))

        return {"same_ortho_diff_tax": {"prot": oprot, "seq": oseq, 
                                        "ortho": oorthos, "tax": otax}, 
                "same_tax_diff_ortho": {"prot": tprot, "seq": tseq, 
                                        "tax": sim_tax, "ortho": tprot_orth}, 
                "ref"                : {"prot": prot,  "seq": rseq, 
                                        "tax": tax, "ortho": orthos}}
    
    def batching(self, results):
        rres = results["ref"]
        ores = results["same_ortho_diff_tax"]
        tres = results["same_tax_diff_ortho"]
        return (rres["prot"], rres["seq"], rres["ortho"], rres["tax"], 
                tres["prot"], tres["seq"], tres["ortho"], tres["tax"],
                ores["prot"], ores["seq"], ores["ortho"], ores["tax"])
        
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            prot      = self.get_from_lmdb(txn, ("prots?", idx))
            for i in range(self.tries):
                results   = self.return_sodt_stdo(txn, prot, 
                                                  topk=self.nearestnghbrs)         
                if results is not None:
                    return self.batching(results)                
        return None
    
    def get_embeddings(self, batches):
        ids, seqs  = zip(*batches)
        lengths    = [len(seq) for seq in seqs]
        maxlen     = max(lengths)
        nbatch     = len(lengths)
        mask       = torch.arange(maxlen, dtype = int).unsqueeze(0).expand(nbatch, maxlen) < torch.tensor(lengths, dtype = int).unsqueeze(1)
        tokens = []
        for b in batches:
            _, _, toks = self.bc([b]) # [1, seqlen]
            tokens.append(toks.squeeze(0))
        tokens       = pad_sequence(tokens, padding_value = 1)
        tokens       = rearrange(tokens, "s b -> b s")
        return tokens, mask
    
    def collatefn(self, batches):
        batches         = list(filter(lambda x : x is not None, 
                                batches))
        print(batches)
        if len(batches) == 0:
            return None
        ( prot,  seq,  ortho,  taxv, 
         tprot, tseq, tortho, ttaxv,
         oprot, oseq, oortho, otaxv) = zip(*batches)
        
        tax             = torch.tensor([self.taxmap[int(t)] for t in  taxv]).long()
        ttax            = torch.tensor([self.taxmap[int(t)] for t in ttaxv]).long()
        otax            = torch.tensor([self.taxmap[int(t)] for t in otaxv]).long()
        
        tokens, mask    = self.get_embeddings(list(zip(prot,   seq)))
        otokens, omask  = self.get_embeddings(list(zip(oprot, oseq)))
        ttokens, tmask  = self.get_embeddings(list(zip(tprot, tseq)))
        return {"reference": {"tokens": tokens, 
                              "mask": mask, 
                              "name": prot, 
                              "seq": seq, 
                              "taxonomy": tax, 
                              "taxids": taxv,
                              "orthoid": ortho},
                "same_tax_different_ortho": {"tokens": ttokens, 
                              "mask": tmask, 
                              "name": tprot, 
                              "seq": tseq, 
                              "taxonomy": ttax, 
                              "taxids": ttaxv,
                              "orthoid": tortho},
                "same_ortho_different_tax": {"tokens": otokens, 
                              "mask": omask, 
                              "name": oprot, 
                              "seq": oseq, 
                              "taxonomy": otax,
                              "taxids": otaxv}}