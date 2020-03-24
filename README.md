# Neural Machine Translation
## What is Machine Translation
Machine translation is the task of automatically converting source text in one language to text in another language.
There are broadly 2 categroies of Machine Translation 
1. Statistical Machine Translation
2. Neural Machine Translation

### Statistical Machine Translation
Statistical machine translation, or SMT for short, is the use of statistical models that learn to translate text from a source language to a target language gives a large corpus of examples.

### Neural Machine Translation
Neural machine translation, or NMT for short, is the use of neural network models to learn a statistical model for machine translation.
The key benefit to the approach is that a single system can be trained directly on source and target text, no longer requiring the pipeline of specialized systems used in statistical machine learning.

This is an attempt to make a encoder decoder neural machine translation architecture.

The file sanity_check.py is taken from Stanford's CSS224n and can be used for running initial checks on encoder decoder and step function
For the Encoder function
``` bash
python sanity_check.py 1d
```
![Image description]()
For the decoder function
``` bash
python sanity_check.py 1e
```
For the step function of Decoder
``` bash
python sanity_check.py 1f
```
To run the code
``` bash
sh run.sh vocab
sh run.sh train_local
```
The file Attention_Encoder_Decoder.ipynb is a concise implementation of the whole process using PyTorch functions like Bucket Iterator and spaCy. The dataset used is the parallel eng-ger corpus in spacy datasets
