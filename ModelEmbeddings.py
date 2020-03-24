#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:35:19 2020

@author: ishan
"""
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
class ModelEmbedding(nn.Module):
  """
  Word to embedding
  Params:
    emb_dim-Embedding Size
    Vocab-Vocabulary 
  """
  def __init__(self,embed_size,vocab):
    super(ModelEmbedding,self).__init__()
    self.embed_size=embed_size

    self.source=None
    self.target=None
    src_pad_token=vocab.src['<pad>']
    tgt_pad_token=vocab.tgt['<pad>']
    
    self.source=nn.Embedding(num_embeddings=len(vocab.src),embedding_dim=embed_size,
                            padding_idx=src_pad_token)
    self.target=nn.Embedding(num_embeddings=len(vocab.tgt),embedding_dim=embed_size,
                            padding_idx=tgt_pad_token)

