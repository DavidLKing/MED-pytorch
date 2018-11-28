#!/usr/bin/env python3

import pdb
import difflib
import pandas as pd

from affixcheck import affixes

a = affixes()

def gen_paradigms(unis):
    paradigms = {}
    for line in unis:
        line = line.strip().split('\t')
        if len(line) > 1:
            assert(len(line) == 3)
            lemma = line[0]
            word = line[1]
            features = line[2]
            if lemma not in paradigms:
                paradigms[lemma] = {}
            if features not in paradigms[lemma]:
                paradigms[lemma][features] = word
    return paradigms

unimorph = open('data/russian/rus-fake-train.tsv', 'r')
vecs = pd.read_csv('russian-w-vecs.tsv', sep='\t', names=['input', 'gold', 'pred'])
novecs = pd.read_csv('russian-no-vecs.tsv', sep='\t', names=['input', 'gold', 'pred'])

paradigms = gen_paradigms(unimorph)

pdb.set_trace()