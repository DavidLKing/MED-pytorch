import sys
import os
import pdb


gold = open(sys.argv[1], 'r').readlines()
pred = open(sys.argv[2], 'r').readlines()

assert(len(gold) == len(pred))

joined_lines = []

# newdoc = None
# newpar = None
text = None

gold_sent = []
pred_sent = []
lemma_sent = []

sent_tripples = []

for gline, pline in zip(gold, pred):
    # if gline.startswith("# newdoc"):
    #     newdoc = gline
    # if gline.startswith("# newpar"):
    #     newpar = gline
    # if gline.startswith("# sent_id"):
    #     sent_id = gline
    if gline.startswith("# text"):
        if gold_sent:
            gold_sent = ' '.join('@'.join(gold_sent))
            pred_sent = ' '.join('@'.join(pred_sent))
            # gold_sent = ' '.join(gold_sent)
            # pred_sent = ' '.join(pred_sent)
            # lemma_sent = ' '.join('@'.join(pred_sent))
            text = ' '.join(text.replace(' ', '@'))
            sent_tripples.append([text, gold_sent, pred_sent]) #, lemma_sent])
        text = gline.strip().replace("# text = ", "")
        gold_sent = []
        pred_sent = []
        lemma_sent = []
    elif gline[0].isnumeric():
        gline = gline.strip().split('\t')
        pline = pline.strip().split('\t')
        # lemma_sent.append(gline[1])
        gold_sent.append(gline[2])
        pred_sent.append(pline[2])

class Vocab():
    def __init__(self):
        self.src = set()
        self.tgt = set()

    def add2vocab(self, sent, vocab):
        for char in sent:
            vocab.add(char)
        # if sent == '':
            # pdb.set_trace()
        # for word in sent:
        #     vocab.add(char)

    def writeout(self, dir_name):
        with open(dir_name + 'vocab.source', 'w') as of:
            for char in self.src:
                of.write(char + '\n')
        with open(dir_name + 'vocab.target', 'w') as of:
            for char in self.tgt:
                of.write(char + '\n')


# full_voacb = Vocab()
p2g_vocab = Vocab()
g2t_vocab = Vocab()

for direction in ['pred2gold', 'gold2text']:
    if not os.path.exists(direction):
        os.makedirs(direction)

# full = open(sys.argv[1] + '.full', 'w')
pred2gold = open('pred2gold/data.txt', 'w')
gold2text = open('gold2text/data.txt', 'w')


for sents in sent_tripples:
    text = sents[0]
    gold = sents[1]
    pred = sents[2]
    p2g_vocab.add2vocab(pred, p2g_vocab.src)
    p2g_vocab.add2vocab(gold, p2g_vocab.tgt)
    g2t_vocab.add2vocab(gold, g2t_vocab.src)
    g2t_vocab.add2vocab(text.split(), g2t_vocab.tgt)
    # gold = ' '.join(gold)
    # pred = ' '.join(pred)
    # bffs = [pred, gold]
    # if len('\t'.join(bffs).split('\t')) != 2:
    #     pdb.set_trace()
    pred2gold.write('\t'.join([pred, gold]) + '\n')
    gold2text.write('\t'.join([gold, text]) + '\n')
    print('pred', pred)
    print('gold', gold)
    print('text', text)

p2g_vocab.writeout('pred2gold/')
g2t_vocab.writeout('gold2text/')