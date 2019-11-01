import sys
import pdb

outfile = open(sys.argv[2], 'w')

for line in open(sys.argv[1], 'r'):
    line = line.strip().split('\t')
    newline = line
    if len(line) > 4:
        newline[2] = newline[1]
        if line[1] in ['人', '他', '她', '它', '我', '牠']:
            if 'Number=Plur' in line[5]:
                newline[2] = line[1] + '們'
        elif len(line[1].split('+')) > 1:
            newline[2] = line[1].replace("+", "")

    outfile.write('\t'.join(newline) + '\n')