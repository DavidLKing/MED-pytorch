#!/usr/bin/env python3

from io import open
import unicodedata
import string
import re
import random
import argparse
import pdb

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

    def buildSentence(self, sequence):
        sequence = sequence.split('\t')
        in_seq = [letters for letters in sequence[0]]
        in_seq += [feats for feats in sequence[1].split(',')]
        out_seq = [letters for letters in sequence[2]]
        return in_seq, out_seq

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

class MED:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--inputs', help="training file (e.g. data/train.txt", required=True)
        parser.add_argument('-v', '--valid', help="validation file (e.g. data/valid.txt", required=True)
        parser.add_argument('-t', '--test', help="testing file (e.g. data/train.txt", required=True)
        self.args = parser.parse_args()
        # print(self.args)
        self.train = Lang('what')
        pdb.set_trace()

    def main(self):
        pass

if __name__ == '__main__':
    m = MED()
    m.main()