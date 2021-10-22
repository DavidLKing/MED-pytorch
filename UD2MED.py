import sys
import pdb

lines = open(sys.argv[1], 'r').readlines()
outfile = open('data.txt', 'w')

outlines = []
input_vocab = set()
output_vocab = set()

for line in lines:
    if not line.startswith("#"):
        line = line.strip().split('\t')
        if len(line) > 6:
            feats = line[5]
            feats = feats.split('|')
            if feats != ['_']:
                # Needed for japanese:
                # if feats == ['_']:
                #     feats = ['None']
                lemma = line[2]
                wordform = line[1]
                pos = line[3]
                input_vocab.add(pos)
                other_pos = line[4]
                input_vocab.add(other_pos)
                head_rel = line[7]
                input_vocab.add(head_rel)

                for char in wordform:
                    output_vocab.add(char)
                for char in lemma:
                    input_vocab.add(char)

                for f in feats:
                    input_vocab.add(f)

                # print(feats)
                # pdb.set_trace()

                feats = ' '.join(feats)
                feats = pos + ' ' + feats
                feats = other_pos + ' ' + feats
                feats = head_rel + ' ' + feats
                wordform = ' '.join(wordform)
                lemma = ' '.join(lemma)

                outfile.write('\t'.join(
                    [feats, lemma, wordform]
                ) + '\n')

with open('vocab.source', 'w') as of:
    for iv in input_vocab:
        of.write(iv + '\n')
with open('vocab.target', 'w') as of:
    for ov in output_vocab:
        of.write(ov + '\n')