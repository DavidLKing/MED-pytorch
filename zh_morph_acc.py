import sys
import pdb

correct = 0
total = 0

for line in open(sys.argv[1], 'r'):
    line = line.strip().split('\t')
    newline = line
    if len(line) > 4:
        total += 1
        newline[2] = newline[1]
        if line[1] in ['人', '他', '她', '它', '我', '牠']:
            if 'Number=Plur' in line[5]:
                newform = line[1] + '們'
        elif len(line[1].split('+')) > 1:
            newform = line[1].replace("+", "")
        else:
            newform = line[1]
        if newform == line[2]:
            correct += 1
        else:
            pdb.set_trace()
print("acc:", correct / total)