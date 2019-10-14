import sys
import pdb

sents = {}

sent_id = None
id_line = None

for line in open(sys.argv[1], 'r'):
    if line.startswith("#sent_id"):
        sent_id = int(line.strip()[11:])
        id_line = line.strip()
    elif line.startswith("#text"):
        sent = line.strip()
        assert(sent_id not in sents)
        sents[sent_id] = [id_line, line]
# pdb.set_trace()

for key in sorted(sents.keys()):
    print(sents[key][0])
    print(sents[key][1])
    # print()