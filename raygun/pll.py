# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import esm
import pandas as pd
import numpy as np
import torch


def get_logits(seq,model,batch_converter,format=None,device=0):
  data = [ ("_", seq),]
  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  batch_tokens = batch_tokens.to(device)
  with torch.no_grad():
      logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()
  if format=='pandas':
    WTlogits = pd.DataFrame(logits[0][1:-1,:],columns=alphabet.all_toks,index=list(seq)).T.iloc[4:24].loc[AAorder]
    WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
    return WTlogits
  else:
    return logits[0][1:-1,:]

def get_PLL(seq,model,alphabet,batch_converter,reduce=np.sum,device=0):
  s=get_logits(seq,model=model,batch_converter=batch_converter,device=device)
  idx=[alphabet.tok_to_idx[i] for i in seq]
  return reduce(np.diag(s[:,idx]))