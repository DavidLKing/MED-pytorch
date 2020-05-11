#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import yaml
import tqdm
import pickle as pkl

# FOR _csv.Error: field larger than field limit (131072)
import csv

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import pdb
import gensim
import numpy as np

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models import TopKDecoder
### Vec mods ###
# from seq2seq.models import VecDecoderRNN
### Done ###
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from micha_condit import cond_prob

# FOR _csv.Error: field larger than field limit (131072)
csv.field_size_limit(sys.maxsize)

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')

ud_t = "data/fi/train/data.txt"
ud_d = "data/fi/dev/data.txt"

srst_t = "/home/david/bin/git/srst18/data19/ru/train/data.txt"
srst_d = "/home/david/bin/git/srst18/data19/ru/dev/data.txt"

def len_filter(example):
    max_len = 50
    return len(example.src) <= max_len and len(example.tgt) <= max_len

def gen_data(train_path, dev_path):
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    train = torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )

    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab
    return train, dev, input_vocab, output_vocab

ud_train, ud_dev, ud_in, ud_out = gen_data(ud_t, ud_d)

# (Pdb) ud_train.examples[0].src
# ['OUT=Animacy=Inan', 'OUT=Case=Nom', 'OUT=Gender=Fem', 'OUT=Number=Sing', 'OUT=NOUN', 'а', 'н', 'к', 'е', 'т', 'а']
# (Pdb) ud_train.examples[0].tgt
# ['<sos>', 'а', 'н', 'к', 'е', 'т', 'а', '<eos>']

srst_train, srst_dev, srst_in, srst_out = gen_data(srst_t, srst_d)

pdb.set_trace()
