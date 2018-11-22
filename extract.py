import sys

'''
extract unimorph data not used in sigmorphon
'''

sig_train = open(sys.argv[1], 'r')
sig_dev = open(sys.argv[2], 'r')
sig_test = open(sys.argv[3], 'r')
unimorph = open(sys.argv[4], 'r')


vocab = set()

def add_vocab(voc_set, sig_data):
    for line in sig_data:
        line = line.strip().split('\t')
        voc_set.add(line[2])
    return voc_set

vocab = add_vocab(vocab, sig_train)
vocab = add_vocab(vocab, sig_dev)
vocab = add_vocab(vocab, sig_test)

print("Total vocab size:", len(vocab))

total_oov = 0
total_in = 0
total = 0
blanks = 0


out = open(sys.argv[5], 'w')

for line in unimorph:
    total += 1
    line = line.split('\t')
    if len(line) > 1:
        if line[1] in vocab:
            total_in += 1
        else:
            total_oov += 1
            out.write('\t'.join(line))
    else:
        blanks += 1

print("Total out of vocab:", total_oov)
print("Total in vocab:", total_in)
print("Total blanks:", blanks)
print("Total:", total)
    
