import sys
import pdb

filename = sys.argv[1]
new_filename = filename[:-4] + '.inv' + filename[-4:]

outfile = open(new_filename, 'w')

for line in open(filename, 'r'):
    line = line.strip().split('\t')
    # inputs = line[0].split(' ')
    outputs = line[2]
    letters = line[1]
    feats = line[0].split(' ')
    # feats = [x for x in inputs if len(x) > 1]
    # letters = [x for x in inputs if len(x) == 1]
    feats.reverse()
    # new_inputs = ' '.join(feats + letters)
    new_line = '\t'.join([' '.join(feats), letters, outputs]) + '\n'
    outfile.write(new_line)

outfile.close()
