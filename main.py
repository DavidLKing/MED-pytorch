#!/usr/bin/env python3

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<S>", 1: "</S>", 2: "<UNK>"}
        self.n_words = 3  # Count <S>, </S>, and <UNK>

    def buildSentence(self, seq_x):
        seq_x = seq_x.split('\t')
        seq = [letters for letters in seq_x[0]]
        if len(seq_x) > 1:
            seq += [feats for feats in seq_x[1].split(',')]
        return seq

    def addSentence(self, sentence):
        sentence = self.buildSentence(sentence)
        # for word in sentence.split(' '):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1