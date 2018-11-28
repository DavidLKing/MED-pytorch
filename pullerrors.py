import sys
import pdb

skipped = 0

for line in open(sys.argv[1], 'r'):
    line = line.strip().split('\t')
    if len(line) == 3:
        if line[1] != line[2]:
            print('\t'.join(line))
    else:
        skipped += 1

# print("skipped", skipped, "lines")