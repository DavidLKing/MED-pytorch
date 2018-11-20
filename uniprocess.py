import numpy
import sys
import pdb
from os.path import basename

class Rearrange():
    def __init__(self, raw_input):
        self.char_vocab_source = {}
        self.char_vocab_target = {}
        # self.char_vocab_source['<S>'] = 0
        # self.char_vocab_source['</S>'] = 2
        # self.char_vocab_source['<UNK>'] = 1
        # self.char_vocab_target['<S>'] = 0
        # self.char_vocab_target['</S>'] = 2
        # self.char_vocab_target['<UNK>'] = 1
        self.morph_vocab = set()
        # self.base = basename(raw_input).split('-')[2]
        # self.base_full = basename(raw_input)
        # vecs = vectors(vecfile)
        # self.model = vecs.get_model()
        # self.vec_input = []

    def load_lines(self, raw_input):
        """
        build character vocabs a preprocess the lines for mila-blocks
        also check for OOVs in the russian word vectors
        :param raw_input:
        :return:
        """
        in_vocab = 0
        total = 0
        lines = []
        source_index = 3
        targ_index = 3
        for presplit in open(raw_input, 'r').readlines():
        # for presplit in open(raw_input, 'r'):
            line = presplit.split('\t')
            if len(line) > 1:
                lemma = line[0]
                for char in lemma:
                    # char = char.strip().encode('utf-8')
                    char = char.strip()
                    if char not in self.char_vocab_source:
                        self.char_vocab_source[char] = source_index
                        source_index += 1
                # erroneous variable name, I just found out that Faruqui's code does not factor the grammar features
                # TODO factor features? Leave them as is? Store both?
                # factored_gram = line[1].split(',')
                # for feature in factored_gram:
                #     self.morph_vocab.add(feature)
                # nonfactored grammatical features
                try:
                    gram_feats = line[2].replace(';', ' ')
                except:
                    pdb.set_trace()
                self.morph_vocab.add(gram_feats)
                wordform = line[1]
                for char in wordform:
                    # char = char.strip().encode('utf-8')
                    char = char.strip()
                    if char not in self.char_vocab_target:
                        self.char_vocab_target[char] = targ_index
                        targ_index += 1
                out = 'OUT='
                featset = []
                for feat in line[2].replace(';', ' ').split():
                    feat = out + feat
                    feat = feat.strip() # .encode('utf-8')
                    # featset.append(feat.encode('utf-8'))
                    featset.append(feat)
                    if feat not in self.char_vocab_source:
                        # self.char_vocab_source[feat.encode('utf-8')] = source_index
                        self.char_vocab_source[feat] = source_index
                        source_index += 1
                total += 1
                l = [line[0], ' '.join(featset), line[1].strip()]
                lines.append(l)
        return lines


    def realign(self, lines):
        outlines = []
        for line in lines:
            assert(len(line) == 3)
            inform = ''
            outform = ''
            # inform += '<S> '
            # build features
            inform += line[1]
            # extra space
            inform += ' '
            # build lemma
            inform += ' '.join(line[0])
            outform += ' '.join(line[2].strip())
            outlines.append((inform, outform))
        return outlines

    def writeout(self, outlines):
        outfile = open('data.txt', 'w')
        for in_out_tuple in outlines:
            inline = in_out_tuple[0]
            outline = in_out_tuple[1] + '\n'
            outfile.write('\t'.join([inline, outline]))
        srcfile = open('vocab.source', 'w')
        tgtfile = open('vocab.target', 'w')
        for char in self.char_vocab_source:
            srcfile.write(char + '\n')
        for char in self.char_vocab_target:
            tgtfile.write(char + '\n')
        print("source vocab length", len(self.char_vocab_source))
        print("target vocab length", len(self.char_vocab_target))

if __name__ == '__main__':
    # if len(sys.argv) < 3:
    #     sys.exit("Please run the program like so,\n"
    #              "python2 mila-blocks-input.py vectors.bin russian-task1-train")
    # print('arg_list', sys.argv)
    # for arg in range(len(sys.argv)):
    #     print(arg, "is", sys.argv[arg])
    # for arg in sys.argv[2:]:
    print("Usage: python3 preprocess.py unimorph german-task1-train")
    print("or: python3 preprocess.py sigmorphon german-task1-train")
    r = Rearrange(sys.argv[1])
    lines = r.load_lines(sys.argv[1])
    outlines = r.realign(lines)
    r.writeout(outlines)
