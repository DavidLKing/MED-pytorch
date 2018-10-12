#!/usr/bin/env python3

import sys
import pdb


print("Sample usage: ./opennmtTest.py sigmorphtrain sigmorphfile source.txt targ.txt")

featset = set()
featdata = open(sys.argv[1], 'r').readlines()
data = open(sys.argv[2], 'r').readlines()
infile = open(sys.argv[3], 'w')
outfile = open(sys.argv[4], 'w')

def featvec(feats, featlist):
    fvec = []
    feats = feats.split(',')
    for f in featlist:
        if f in feats:
            f += "=1"
            # fvec.append(f += "=1")
            # fvec.append(featlist.index(f))
        else:
            f += "=0"
            # fvec.append('='.join([f.split('=')[0], str(0)]))
        fvec.append(f)
    assert(len(fvec) == len(featlist))
    return fvec

def srcformat(feats, src):
    outdata = []
    for char in src:
        char += "|"
        char += "|".join([str(x) for x in feats])
        outdata.append(char)
    return ' '.join(outdata)

for line in featdata:
    line = line.strip().split('\t')
    feats = line[1]
    for f in feats.split(','):
       featset.add(f)

# featset = list(featset)
featset = sorted(featset)

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
