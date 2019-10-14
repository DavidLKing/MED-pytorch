#!/usr/bin/env python3

import sys
import pdb
import gensim
import difflib
import argparse

from affixcheck import affixes

a = affixes()

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--embeddings',
                    type=str,
                    required=True,
                    default=None,
                    help='Word embeddings file (binary)')
parser.add_argument('-no', '--noun_output',
                    type=str,
                    required=True,
                    default=None,
                    help='Noun output file')
parser.add_argument('-vo', '--verb_output',
                    type=str,
                    required=True,
                    default=None,
                    help='Verb output file')
parser.add_argument('-udt', '--unidep_train',
                    type=str,
                    required=True,
                    default=None,
                    help='Universal Dependency training file')
parser.add_argument('-udd', '--unidep_dev',
                    type=str,
                    required=True,
                    default=None,
                    help='Universal Dependency development file')
parser.add_argument('-ude', '--unidep_eval',
                    type=str,
                    required=True,
                    default=None,
                    help='Universal Dependency testing file')
parser.add_argument('-uni', '--unimorph',
                    type=str,
                    required=True,
                    default=None,
                    help='Unimorph file')

opt = parser.parse_args()

vec_file = opt.embeddings
ud_train = opt.unidep_train
ud_dev = opt.unidep_dev
ud_test = opt.unidep_eval

unimorph = open(opt.unimorph, 'r')

# Feel free to hard code stuff too if that's easier for you
# vec_file = 'ruscorpora.model.bin'
# ud_train = './UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu'
# ud_dev = './UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu'
# ud_test = './UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu'
#
# unimorph = open('data/russian/uni/rus-fake-train.tsv', 'r')


# Paradigm extraction

def gen_paradigms(unis):
    paradigms = {}
    for line in unis:
        line = line.strip().split('\t')
        if len(line) > 1:
            assert(len(line) == 3)
            # TODO fix space bug
            # if lemma == 'геркулесоваякаша':
            # should be геркулесовая каша
            #     pdb.set_trace()
            lemma = line[0].replace(' ', '')
            word = line[1].replace(' ', '')
            features = line[2]
            if lemma not in paradigms:
                paradigms[lemma] = {}
            if features not in paradigms[lemma]:
                paradigms[lemma][features] = word
    return paradigms



paradigms = gen_paradigms(unimorph)

print("loading word vectors")
vecs = gensim.models.KeyedVectors.load_word2vec_format(vec_file, binary=True)
print("loading UDs")



################## VERB CLASS STUFF ###########################
def get_verb_class(cite, paradigms):

    if cite in paradigms:
        inform = 'V;PRS;2;SG'
        if inform in paradigms[cite]:
            second_sing = paradigms[cite][inform]
            _, _, _, affixes = a.diffasstring(cite, second_sing)
            for affix in affixes:
                if '+е' in affix:
                    conj_class = 'e-conj'
                elif '+ё' in affix:
                    conj_class = 'ee-conj'
                else:
                    conj_class = 'i-conj'
            return conj_class
        else:
            return None
    else:
        return None

################################################################

def load_UDs(ud_data, vecs, paradigms):
    nouns = set()
    verbs = set()
    for line in open(ud_data, 'r').readlines():
        if not line.startswith("#"):
            line = line.lower().strip().split('\t')
            if len(line) >= 6:
                feats = line[5]
                lemma = line[2]
                wordform = line[1]
                if line[3] == 'noun':
                    feats = feats.split("|")
                    if feats[0].startswith('animacy'):
                        if lemma in vecs:
                            vec_string = ' '.join([str(x).strip() for x in vecs[lemma]])
                            noun_class = feats[0].split('=')[1]
                            nouns.add((noun_class, vec_string))
                elif line[3] == 'verb':
                    if lemma in paradigms and lemma in vecs:
                        verb_class = get_verb_class(lemma, paradigms)
                        if verb_class:
                            vec_string = ' '.join([str(x).strip() for x in vecs[lemma]])
                            verbs.add((verb_class, vec_string))

    return nouns, verbs

def write_data(data, file):
    for datum in data:
        line = ' '.join(list(datum)) + '\n'
        file.write(line)

print("Starting train")
train_nouns, train_verbs = load_UDs(ud_train, vecs, paradigms)
print("Starting dev")
dev_nouns, dev_verbs = load_UDs(ud_dev, vecs, paradigms)
print("Starting test")
test_nouns, test_verbs = load_UDs(ud_test, vecs, paradigms)

# pdb.set_trace()

# train_file = 'train.txt'
# dev_file = 'dev.txt'
# test_file = 'test.txt'

print("Writing files")

with open(opt.noun_output, 'w') as of:
    of.write("TRAIN\n")
    write_data(train_nouns, of)
    of.write("DEV\n")
    write_data(dev_nouns, of)
    of.write("TEST\n")
    write_data(test_nouns, of)

with open(opt.verb_output, 'w') as of:
    of.write("TRAIN\n")
    write_data(train_verbs, of)
    of.write("DEV\n")
    write_data(dev_verbs, of)
    of.write("TEST\n")
    write_data(test_verbs, of)
