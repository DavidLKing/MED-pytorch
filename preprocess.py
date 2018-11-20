import numpy as np
import pickle as pkl
import sys
import gensim
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
        self.base = basename(raw_input).split('-')[2]
        self.base_full = basename(raw_input)
        # vecs = vectors(vecfile)
        # self.model = vecs.get_model()
        # self.vec_input = []

    def load_lines(self, feat_struct, raw_input):
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
            lemma = line[0]
            for char in lemma:
                # char = char.strip().encode('utf-8')
                char = char.strip()
                if char not in self.char_vocab_source:
                    self.char_vocab_source[char] = source_index
                    source_index += 1
            if len(line) > 1: 
                if feat_struct == 'sigmorphon':
                    wordform = line[2].strip()
                    feats = line[1].split(',')
                    feats = [f.split('=')[1] for f in feats]
                elif feat_struct == 'unimorph':
                    wordform = line[1]
                    feats = line[2].strip().split(';')
                else:
                    sys.exit("must select unimorph or sigmorphon for feature structure")
                for char in wordform:
                    char = char.strip()
                    if char not in self.char_vocab_target:
                        self.char_vocab_target[char] = targ_index
                        targ_index += 1
                out = 'OUT='
                featset = []
                # ADD FEATS TO INPUT
                for feat in feats:
                    feat = out + feat
                    feat = feat.strip() 
                    featset.append(feat)
                    if feat not in self.char_vocab_source:
                        self.char_vocab_source[feat] = source_index
                        source_index += 1
                total += 1
                l = [lemma, ' '.join(featset), wordform]
                lines.append(l)
            else:
                print("line", line, "ignored")
        return lines


    def realign(self, lines, vecs):
        outlines = []
        found_vecs = []
        found = 0
        total = 0
        for line in lines:
            total += 1
            assert(len(line) == 3)
            inform = ''
            outform = ''
            # inform += '<S> '
            # build features
            inform += line[1]
            # extra space
            inform += ' '
            # build lemma
            lemma = line[0]
            inform += ' '.join(lemma)
            outform += ' '.join(line[2])
            if vecs:
                vec_size = vecs.vector_size
                if lemma in vecs:
                    found += 1
                    found_vecs.append(vecs[lemma])
                else:
                    found_vecs.append(np.random.normal(0.0, 0.1, vec_size))
            outlines.append((inform, outform))
        print("Found", found, "of", total, "word vectors")
        print("Coverate:", found / total)
        # pdb.set_trace()
        if vecs:
            assert(len(outlines) == len(found_vecs))
            return outlines, found_vecs
        else:
            return outlines, None

    def writeout(self, outlines, vecs):
        outfile = open('data.txt', 'w')
        for in_out_tuple in outlines:
            inline = in_out_tuple[0]
            outline = in_out_tuple[1] + '\n'
            outfile.write('\t'.join([inline, outline]))
        if self.base == 'train':
            srcfile = open('vocab.source', 'w')
            tgtfile = open('vocab.target', 'w')
            for char in self.char_vocab_source:
                srcfile.write(char + '\n')
            for char in self.char_vocab_target:
                tgtfile.write(char + '\n')
            print("source vocab length", len(self.char_vocab_source))
            print("target vocab length", len(self.char_vocab_target))
        if vecs:
            print("Writing out vectors:")
            pkl.dump(vecs, open('vectors.pkl', 'wb'))
        print("finished")

if __name__ == '__main__':
    # if len(sys.argv) < 3:
    #     sys.exit("Please run the program like so,\n"
    #              "python2 mila-blocks-input.py vectors.bin russian-task1-train")
    # print('arg_list', sys.argv)
    # for arg in range(len(sys.argv)):
    #     print(arg, "is", sys.argv[arg])
    # for arg in sys.argv[2:]:
    print("Usage: python3 preprocess.py unimorph german-task1-train [vectors]")
    print("or: python3 preprocess.py sigmorphon german-task1-train [vectors]")
    if 4 <= len(sys.argv):
        print("Found word vectors in", sys.argv[3], "\nLoading...")
        vecs = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[3], binary=True)
        print("Loaded")
    else:
        vecs = None
    r = Rearrange(sys.argv[2])
    lines = r.load_lines(sys.argv[1], sys.argv[2])
    outlines, vecs = r.realign(lines, vecs)
    r.writeout(outlines, vecs)
