import sys

for line in open(sys.argv[1], 'r'):
    line = line.strip().split('\t')
    if line[1] != line[2]:
        print('\t'.join(line))
