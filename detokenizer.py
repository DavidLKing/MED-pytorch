import sys
import pdb
from nltk.tokenize.treebank import TreebankWordDetokenizer

detok = lambda x: TreebankWordDetokenizer().detokenize(x.split())

inputfile = open(sys.argv[1], 'r').readlines()

# pdb.set_trace()

for line in inputfile:
    if line.startswith("#text = "):
        detoked_line = detok(line[8:])
        print("#text = " + detoked_line.strip())
    else:
        print(line.strip())