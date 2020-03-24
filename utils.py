import torch
import torch
##torch.cuda.current_device()
#torch.cuda.device(0)
#torch.cuda.device_count()
#torch.cuda.get_device_name(0)


import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')

def pad_sents(sents,pad_tokens):
  """
  Pad the sentence wrt to the largest in the batch
  Params
    sents is a batch of sentences
    pad_tokens is the token to place
  """

  sents_padded=[]
  sent_padded=[]
  max_len=len(sents[0])
  for sent in sents:
    max_len=max(len(sent),max_len)


  print(max_len)
  for sent in sents:
    #Append sent+(max_len-len(sent))*[pad_tokens] for each sentence
    sents_padded.append(sent+(max_len-len(sent))*[pad_tokens])


  return sents_padded

testing_padded_seq=[["Hello","My","Name"],
   ["Is","Ishan"],
   ["I","am","a","student"]]

#print(pad_sents(testing_padded_seq,'#'))
               
               
def read_corpus(file_path,source):
  """
  Read sentences which are terminated by newline
  Params:
    Path File-path to file containing corpus
    source-tgt or src denoting if it is source or target
  """
  data=[]
  for line in open(file_path):
    sent=nltk.word_tokenize(line)

    if source=='tgt':
      sent=['<s>']+sent+['</s>']
    data.append(sent)
  return data              


def batch_iter(data,batch_size,shuffle=False):
  """Gives batchs of Data
  Normally use bucketIterator
  Params:
    data-Data Corpus 
    batch_size-Batch Size
    shuffle-Bool whether randomly shuffle the data"""

  num_batchs=math.ceil(len(data)/batch_size)
  index_array=list(range(num_batchs))
  if shuffle:
    np.random.shuffle(index_array)
  
  for i in range(num_batchs):
    ##ith batch consists of sentences from i*batch_size to i+1
    indices=index_array[i*batch_size:(i+1)*batch_size]
    examples=[data[idx] for idx in indices]
    examples=sorted(examples, key=lambda e:len(e[0]),reverse=True)
    src_sents=[e[0] for e in examples]
    tgt_sents=[e[1] for e in examples]

    yield src_sents, tgt_sents