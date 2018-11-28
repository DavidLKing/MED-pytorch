import sys
from random import shuffle

'''
Load unimorph data not in sigmorphon 2016, shuffle, and make test set
Don't know how to do this without loading all the lines at once
'''

unimorphs = open(sys.argv[1], 'r').readlines()
outfile = open(sys.argv[2], 'w')

shuffle(unimorphs)

for line in unimorphs:
    outfile.write(line)
    # line = line.strip().split('\t')
    # lemma = list(line[0])
    # form = list(line[1])
    # feats = line[2].split(';')
    # inputs = feats + lemma
    # print(' '.join(inputs), '\t', ' '.join(form))
