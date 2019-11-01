import sys
import pdb

output = open(sys.argv[1], 'r').readlines()

cased_correct = 0
uncased_correct = 0

total = 0
for line in output:
    line = line.strip().split('\t')
    assert(len(line) == 3)
    if line[1] == line[2]:
        cased_correct += 1
    if line[1].lower() == line[2].lower():
        uncased_correct += 1
        # cased_correct += 1
    # else:
    #     pdb.set_trace()
    total += 1

print("cased acc:", cased_correct / total)
print("uncased acc:", uncased_correct / total)