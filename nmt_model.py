#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:36:04 2020

@author: ishan
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from ModelEmbeddings import ModelEmbedding
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """NMT Model consisting of
    -BiLSTM Encoder
    -Unidir LSTM Decoder
    -Global Attention Model
    """
    def __init__(self,embed_size,hidden_size,vocab,dropout_rate=0.5):
        super(NMT,self).__init__()
        self.model_embeddings = ModelEmbedding(embed_size, vocab)
        self.hidden_size=hidden_size
        self.dropout=dropout_rate
        self.vocab=vocab

        #Default Values
    

        print("HELLO INIT")

        self.encoder=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1,
                         bidirectional=True,dropout=self.dropout)
        self.decoder=nn.LSTMCell(input_size=embed_size+hidden_size,hidden_size=hidden_size)
        self.h_reduction=nn.Linear(2*self.hidden_size,self.hidden_size,bias=False)
        self.c_reduction=nn.Linear(2*self.hidden_size,self.hidden_size,bias=False)

        self.for_attention_h_red=nn.Linear(2*self.hidden_size,hidden_size,bias=False)

        self.final_predictor=nn.Linear(3*self.hidden_size,self.hidden_size,bias=False)
        self.vocab_distribution=nn.Linear(self.hidden_size,self.model_embeddings.target.weight.shape[0])
        self.dropout=nn.Dropout(self.dropout)


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """Takes a batch of Source and Target, computes the log likelihood
    of target sentences under the language models

    Params-
      source-List of Source sentences
      target-List of target sentences wrapped by<start> and </end>

      scores-A variable/Tensor of shape(b,) representing log likelihood"""
      
        source_len=[len(s) for s in source]
        source_padded=self.vocab.src.to_input_tensor(source,device=self.device)
        ##Tensor of Dim (src_len,batch_size)
        target_padded=self.vocab.tgt.to_input_tensor(target,device=self.device)
        ##Tensor of Dim (tgt_len,batch_size)
        encoded_sent,last_hid=self.encode(source_padded,source_len)
        enc_masks=self.generate_sent_masks(encoded_sent,source_len)
        combined_output=self.decode(encoded_sent,enc_masks,last_hid,target_padded)
        P=F.log_softmax(self.vocab_distribution(combined_output),dim=-1)
        #? USE
        target_masks=(target_padded!=self.vocab.tgt['<pad>']).float()
        #computing Log Prob
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self,source_padded:torch.Tensor,source_lengths:List[int])->Tuple[torch.tensor,Tuple[torch.Tensor,torch.Tensor]]:
        """Encoder representation of Source sentence
        Params-
        1. source_padded(src_len,batch_size)
        2. source lengths-List of actual Length of the sentences
        """
        
        X=self.model_embeddings.source(source_padded)
        #X dim should be(srclen, batch, embeddingsize)
        #X=pack_padded_sequence(X,source_lengths)
        enc_hidden,(last_hid,last_cell)=self.encoder(pack_padded_sequence(X,source_lengths))
        
        enc_hidden=pad_packed_sequence(enc_hidden,batch_first=True)[0]
        dec_hidden=torch.cat((last_hid[0,:],last_hid[1,:]),1)
        dec_hidden=self.h_reduction(dec_hidden)
        
        cell_last=torch.cat((last_cell[0,:],last_cell[1,:]),1)
        dec_cell=self.c_reduction(cell_last)
        dec_init_state=(dec_hidden,dec_cell)
      
        return enc_hidden,dec_init_state

    def decode(self,enc_hidden:torch.Tensor,enc_masks:torch.Tensor,
             dec_init_state:Tuple[torch.Tensor,torch.Tensor],target_padded:torch.Tensor)-> torch.Tensor:

        """Combined Output calculate for entire batch
        Params-
        1. enc_hidden-(batch,src_len,hidd*2)
        2.enc_masks:(batch_size,src_len)
        3.dec_init_state-(tuple(Tensor,Tensor)) Initial Hidden and Cell State
        4.target_padded(Tensor)- (tgt_len,batch_size)"""
    
    #End token for longest is chopped off
    
        target_padded=target_padded[:-1]

    ##initialising hidden state
        dec_state=dec_init_state


        batch_size=enc_hidden.size(0)
        o_prev=torch.zeros(batch_size,self.hidden_size,device=self.device)

        combined_output=[]

        enc_hidden_proj=self.for_attention_h_red(enc_hidden)

        y = self.model_embeddings.target(target_padded)

        for y_t in torch.split(y,1):
            y_t=torch.squeeze(y_t)
            ybar_t=torch.cat((y_t,o_prev),dim=1)
            dec_hid,o_t,e_t=self.step(ybar_t,dec_state,enc_hidden,
                                enc_hidden_proj,enc_masks)
            combined_output.append(o_t)
            o_prev=o_t
    
        combined_outputs=torch.stack(combined_output)

        return combined_outputs

    def step(self,ybar_t:torch.Tensor,
           dec_state:Tuple[torch.Tensor,torch.Tensor],
           enc_hiddens:torch.Tensor,enc_hiddens_proj:torch.Tensor,
           enc_masks:torch.Tensor)->Tuple[Tuple,torch.Tensor,torch.Tensor]:

        """
           Computes 1 step of Decoder including Attention
           Params-
            1. ybar_t:concatenated Tensor 
            2. dec_state (tuple(Tensor,Tensor)) prev hidden and cell state
            3. enc_hidden (batch,src_len,h*2)
            4. enc_hidden_proj (batch,src_len,h)
            5. enc_masks Tensor of sentence masks

        """
        dec_state=self.decoder(ybar_t,dec_state)
        dec_hidden,dec_cell=dec_state
        e_t=torch.squeeze(
                torch.bmm(enc_hiddens_proj,torch.unsqueeze(dec_hidden,2)),2)
          
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        alpha_t=F.softmax(e_t,dim=1)  ##attention vector

        a_t=torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t,1),enc_hiddens),1) #Context Vector
        U_t = torch.cat((a_t, dec_hidden), 1)
        V_t = self.final_predictor(U_t)
        O_t = self.dropout(torch.tanh(V_t))

        combined_output=O_t
        return dec_state,combined_output,e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor,
                            source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.
        param 
        1. enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        2. source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        3. enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len), 
        where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(
                enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self,
                    src_sent: List[str],
                    beam_size: int = 5,
                    max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
        value: List[str]: the decoded target sentence, represented as a list of words
        score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var,
                                                  [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(
                len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(
                    hyp_num, src_encodings_att_linear.size(1),
                    src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor(
                    [self.vocab.tgt[hyp[-1]] for hyp in hypotheses],
                    dtype=torch.long,
                    device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(
                    x,
                    h_tm1,
                    exp_src_encodings,
                    exp_src_encodings_att_linear,
                    enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(
                    self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (
                    hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                    prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(
                            Hypothesis(
                                    value=new_hyp_sent[1:-1],
                                    score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(
                    live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                    Hypothesis(
                            value=hypotheses[0][1:], score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    
    
    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device
    
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {'args':
            dict(
                embed_size=self.model_embeddings.embed_size,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate),
            'vocab':self.vocab,
            'state_dict':self.state_dict() }
        torch.save(params, path)
        
        
        
