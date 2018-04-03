#!/usr/bin/env python3

import sys
import pdb


print("Sample usage: ./opennmtTest.py sigmorphfile source.txt targ.txt")

featset = set()
data = open(sys.argv[1], 'r').readlines()

infile = open(sys.argv[2], 'w')
outfile = open(sys.argv[3], 'w')

def featvec(feats, featlist):
    fvec = []
    feats = feats.split(',')
    for f in featlist:
        if f in feats:
            fvec.append(featlist.index(f))
        else:
            fvec.append(0)
    assert(len(fvec) == len(featlist))
    return fvec

def srcformat(feats, src):
    outdata = []
    for char in src:
        char += "|"
        char += "|".join([str(x) for x in feats])
        outdata.append(char)
    return ' '.join(outdata)

for line in data:
    line = line.strip().split('\t')
    feats = line[1]
    for f in feats.split(','):
       featset.add(f)

featset = list(featset)

for line in data:
    line = line.strip().split('\t')
    src = line[0]
    feats = line[1]
    trg = line[2]
    fvec = featvec(feats, featset)
    src = srcformat(fvec, src)
    src += '\n'
    trg = ' '.join(trg) + '\n'
    infile.write(src)
    outfile.write(trg)
