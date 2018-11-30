import pdb
import random
from operator import itemgetter

freqs = open('ancora.freqs.tsv', 'r')

# no need for a dict since they're already sorted
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch/data/spanish-balanced$ cut -f 2 ../../../sigmorphon-childes-balancer/ud-spanish/*.conllu.txt | grep -v "#" | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn > ancora.freqs.tsv
counts = []

for line in freqs:
    line = line.strip().split()
    if len(line) == 2:
        count = int(line[0])
        word = line[1]
        counts.append((count, word))

unis = open('uni.supple.overreg.tsv', 'r')

unimorph = {}
for line in unis:
    line = line.strip().split('\t')
    if len(line) == 3:
        lemma = line[0]
        form = line[1]
        feats = line[2]
        if form not in unimorph:
            unimorph[form] = []
        joint = '\t'.join([lemma, feats])
        if joint not in unimorph[form]:
            unimorph[form].append(joint)

outlines = []
for pair in counts:
    count = pair[0]
    form = pair[1]
    if form in unimorph:
        freq = 0
        while freq < count:
            match = random.choice(unimorph[form])
            match = match.split('\t')
            lemma = match[0]
            feats = match[1]
            outlines.append('\t'.join([lemma, form, feats])+ '\n')
            freq += 1

random.shuffle(outlines)
outfile = open('uni.supple.overreg.balanced.tsv', 'w')
for line in outlines:
    outfile.write(line)
