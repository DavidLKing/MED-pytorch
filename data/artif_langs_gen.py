#!/bin/env/env python3

import pdb
import random
import os
import tqdm

class GenLangs:
    def __init__(self):
        """
        Structures pulled from Baerman et al., 2019.

        Structures:
        Hierarchical
          A B C D
        1 a a b b
        2 c d e f
        3 g g g g

        Cross-classifying
          A B C D
        1 a a b b
        2 c d c d
        3 e f f e

        Grid
          A B C D
        1 a b c d
        2 e f g h
        3 i j k l
        """

        self.prefixes = {
            '1': 'OUT=1',
            '2': 'OUT=2',
            '3': 'OUT=3'
        }

        self.hier_classes = {
            'A': ['a', 'c', 'g'],
            'B': ['a', 'd', 'g'],
            'C': ['b', 'e', 'g'],
            'D': ['b', 'f', 'g']
        }

        self.cross_classes = {
            'A': ['a', 'c', 'e'],
            'B': ['a', 'd', 'f'],
            'C': ['b', 'c', 'f'],
            'D': ['b', 'd', 'e']
        }

        self.grid_classes = {
            'A': ['a', 'e', 'i'],
            'B': ['b', 'f', 'j'],
            'C': ['c', 'g', 'k'],
            'D': ['d', 'h', 'l']
        }

        self.faux_russian_nouns = {
            'animate': ['a', 'a', 'i'],
            'inanimate': ['a', 'i', 'i']
        }

        # Take from Stump and Finkel, chapter 7, page 182, table 7.1

        latin_para = lambda n, g, d, a, v, ab, l: {'nom': n, 'gen': g, 'dat': d, 'acc': a, 'abl': ab, 'loc': l}

        self.real_latin_nouns = {
            'first' : 'placeholder'
        }

        self.faux_latin = {
            'sg nom': 'a',
            'sg gen': 'b',
            'sg dat': 'c',
            'pl nom': 'd',
            'pl gen': 'e',
            'pl dat': 'f'
        }

        # These are wrong
        # self.faux_latin_verbs = {
        #     'PrPrIAc1s': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'c', 'a', 'd', 'c', 'c', 'c', 'c'],
        #     'PrPrIAc2s': ['e', 'e', 'e', 'f', 'f', 'f', 'f', 'f', 'g', 'g', 'g', 'g', 'g', 'h', 'h', 'g', 'g', 'g', 'g'],
        #     'PrPrIAc3s': ['e', 'e', 'e', 'f', 'f', 'f', 'f', 'f', 'i', 'i', 'i', 'i', 'i', 'h', 'h', 'g', 'g', 'g', 'g'],
        #     'PrPrIAc1p': ['e', 'e', 'e', 'f', 'f', 'f', 'f', 'f', 'i', 'i', 'i', 'i', 'i', 'i', 'j', 'g', 'g', 'g', 'g'],
        #     'PrPrIAc2p': ['e', 'e', 'e', 'f', 'f', 'f', 'f', 'f', 'i', 'i', 'i', 'i', 'i', 'h', 'h', 'g', 'g', 'g', 'g'],
        #     'PrPrIAc3p': ['e', 'e', 'e', 'f', 'f', 'f', 'f', 'f', 'j', 'j', 'j', 'j', 'k', 'j', 'j', 'k', 'k', 'k', 'k']
        # }
        # ō	ō	ō	eō	eō	eō	eō	eō	ō	ō	ō	ō	iō	ō	um	iō	iō	iō	iō
        # ā	ā	ā	ē	ē	ē	ē	ē	i	i	i	i	i	∅	∅	ī	ī	ī	ī
        # ā	ā	ā	ē	ē	ē	ē	ē	i	i	i	i	i	∅	∅	ī	ī	ī	ī
        # ā	ā	ā	ē	ē	ē	ē	ē	i	i	i	i	i	i	u	ī	ī	ī	ī
        # ā	ā	ā	ē	ē	ē	ē	ē	i	i	i	i	i	∅	∅	ī	ī	ī	ī
        # a	a	a	e	e	e	e	e	u	u	u	u	iu	u	u	iu	iu	iu	iu

        '''
        % present present indicative active 
        CONJ     PrPrIAc1s PrPrIAc2s PrPrIAc3s  PrPrIAc1p PrPrIAc2p PrPrIAc3p % PrPrIAc3p   
        TEMPLATE 4A        1As       1At        4Amus     1Atis     4Ant       % 4Aunt        
        cIa      ō         ā         ā          ā         ā         a       % ā           
        cIb      ō         ā         ā          ā         ā         a       % ā           
        cIc      ō         ā         ā          ā         ā         a       % ā           
        cIIa     eō        ē         ē          ē         ē         e       % ē           
        cIIb     eō        ē         ē          ē         ē         e       % ē           
        cIIc     eō        ē         ē          ē         ē         e       % ē           
        cIId     eō        ē         ē          ē         ē         e       % ē           
        cIIe     eō        ē         ē          ē         ē         e       % ē           
        cIIIa    ō         i         i          i         i         u       % ∅            
        cIIIb    ō         i         i          i         i         u       % ∅            
        cIIIc    ō         i         i          i         i         u       % ∅            
        cIIId    ō         i         i          i         i         u       % ∅            
        cIIIe    iō        i         i          i         i         iu      % i           
        cIIIf    ō         ∅         ∅          i         ∅         u       % ∅           
        cIIIs    um        ∅         ∅          u         ∅         u       % ∅          
        cIVa     iō        ī         ī          ī         ī         iu      % ī           
        cIVb     iō        ī         ī          ī         ī         iu      % ī           
        cIVc     iō        ī         ī          ī         ī         iu      % ī           
        cIVd     iō        ī         ī          ī         ī         iu      % ī 
        '''

        # For simplicity, all variable names have same affix onset except for vowels
        # a -> q, e -> p, and i -> m---don't to avoid CVCV structure mismatch
        self.affixes = {
            'a': 'qi',
            'b': 'bi',
            'c': 'ci',
            'd': 'di',
            'e': 'pi',
            'f': 'fi',
            'g': 'gi',
            'h': 'hi',
            'i': 'mi',
            'j': 'ji',
            'k': 'ki',
            'l': 'li'
        }

        self.hier_affixes = {
            'a': 'qi',
            'b': 'bi',
            'c': 'ci',
            'd': 'kad',
            'e': 'kap',
            'f': 'kaf'
        }

        self.hier_invert_affixes = {
            'a': 'qi',
            'b': 'bi',
            'c': 'ci',
            'd': 'qid',
            'e': 'bip',
            'f': 'cif'
        }

        self.hier_invert_affixes_full = {
            'a': 'qis',
            'b': 'bis',
            'c': 'cis',
            'd': 'dip',
            'e': 'fip',
            'f': 'gip'
        }

        self.case_num_affixes = {
            'a': 'qisi',
            'b': 'bisi',
            'c': 'cisi',
            'd': 'qipi',
            'e': 'bipi',
            'f': 'cipi'
        }

        self.num_case_affixes = {
            'a': 'siqi',
            'b': 'sibi',
            'c': 'sici',
            'd': 'piqi',
            'e': 'pibi',
            'f': 'pici'
        }

        self.case_num_affixes = {
            'a': 'qisi',
            'b': 'bisi',
            'c': 'cisi',
            'd': 'qipi',
            'e': 'bipi',
            'f': 'cipi'
        }

        self.case_num_1_case_affixes = {
            'a': 'viqi',
            'b': 'tibi',
            'c': 'wici',
            'd': 'yiqi',
            'e': 'zibi',
            'f': 'rici'
        }

        self.case_num_1_num_affixes = {
            'a': 'visi',
            'b': 'tisi',
            'c': 'wisi',
            'd': 'yipi',
            'e': 'zipi',
            'f': 'ripi'
        }

        self.case_case_num_2_affixes = {
            'a': 'qiwi',
            'b': 'biti',
            'c': 'civi',
            'd': 'qiyi',
            'e': 'biri',
            'f': 'cizi'
        }

        self.num_case_num_2_affixes = {
            'a': 'siwi',
            'b': 'siti',
            'c': 'sivi',
            'd': 'piyi',
            'e': 'piri',
            'f': 'pizi'
        }



        self.consonants = [
            'b', 'c', 'd', 'f', 'g', 'h', 'j',
            'k', 'l', 'm', 'n', 'p', 'q', 'r',
            's', 't', 'v', 'w', 'x', 'y', 'z'
        ]

        self.vowel = ['a', 'e', 'i', 'o', 'u']

        # self.vocab = set(self.vowel + self.consonants + list(self.prefixes.values()) + list(self.grid_classes.keys()))
        self.vocab = set(self.vowel + self.consonants)
        self.feat_vocab = list(self.prefixes.values())

    def gen_pref(self, form_list):
        prefixes = []
        for i, form in enumerate(form_list):
            prefixes.append("OUT={}".format(i+1))
        return prefixes

    def gen_stems(self):
        stems = set()
        for cons1 in self.consonants:
            for vow1 in self.vowel:
                for cons2 in self.consonants:
                    for vow2 in self.vowel:
                        stems.add(cons1 + vow1 + cons2 + vow2)
        return stems

    def gen_paradigm(self, prefixes, classes, stem, affixes, which_class):
        paradigm = []
        struct = classes[which_class]
        assert(len(prefixes) == len(struct))
        # sorted to prevent mutable misordering
        for slot, cell in zip(sorted(prefixes), struct):
            affix = affixes[cell]
            try:
                pref = prefixes[slot]
            except:
                # pdb.set_trace()
                pref = slot
            # input = pref + ' OUT=' + which_class + ' ' + ' '.join(stem)
            input = pref + ' ' + ' '.join(stem + affix)
            # output = ' '.join(stem) + ' ' + ' '.join(affix)
            paradigm.append(input)
        lines = []
        # for para in paradigm:
        #     for idx in range(3):
        #
        for p1 in paradigm:
            for p2 in paradigm:
                for p3 in paradigm:
                    if p1 != p2 and p2 != p3 and p3 != p1:
                        feats = [x for x in p2.split(' ') if len(x) > 1]
                        form = [x for x in p2.split(' ')  if len(x) == 1]
                        in_1 = p1.replace("OUT=", "IN=")
                        in_3 = p3.replace("OUT=", "IN=")
                        outline = ' '.join(feats) + ' ' + in_1 + ' ' + in_3 + '\t' + ' '.join(form) + '\n'
                        lines.append(outline)
        return lines

    def writeout(self, data, dir_name):
        # set up 70, 10, 20 split
        item_nums = len(data) // 10
        train_idxes = item_nums * 7
        dev_idxes = train_idxes + item_nums
        train_set = data[0:train_idxes]
        dev_set = data[train_idxes:dev_idxes]
        test_set = data[dev_idxes:]

        random.shuffle(train_set)
        random.shuffle(dev_set)
        random.shuffle(test_set)

        for dir, dataset in zip(
            [dir_name, dir_name + '/train', dir_name + '/dev/', dir_name + '/test'],
            [data, train_set, dev_set, test_set]
        ):
            if not os.path.exists(dir):
                os.mkdir(dir)
            with open(dir + '/' + 'data.txt', 'w') as of:
                for line in dataset:
                    of.write(line)
            with open(dir + '/' + 'vocab.source', 'w') as vs:
                with open(dir + '/' + 'vocab.target', 'w') as vt:
                    for elem in self.vocab:
                        vs.write(elem + '\n')
                        vt.write(elem + '\n')
            with open(dir + '/' + 'vocab.feats', 'w') as vf:
                for elem in self.feat_vocab:
                    vf.write(elem + '\n')

    def hier_writeout(self, data, dir_name):
        # set up 70, 10, 20 split
        item_nums = len(data) // 10
        train_idxes = item_nums * 7
        dev_idxes = train_idxes + item_nums
        train_set = data[0:train_idxes]
        dev_set = data[train_idxes:dev_idxes]
        test_set = data[dev_idxes:]

        random.shuffle(train_set)
        random.shuffle(dev_set)
        random.shuffle(test_set)

        vocab = set()
        feat_vocab = set()
        for entry in data:
            entry = entry.split('\t')
            ins = entry[1].split(' ')
            outs = entry[2].split(' ')
            feats = entry[0].split(' ')
            for elem in ins + outs:
                vocab.add(elem)
            for elem in feats:
                feat_vocab.add(elem)

        vocab = sorted(vocab)

        for dir, dataset in zip(
            [dir_name, dir_name + '/train', dir_name + '/dev/', dir_name + '/test'],
            [data, train_set, dev_set, test_set]
        ):
            if not os.path.exists(dir):
                os.mkdir(dir)
            with open(dir + '/' + 'data.txt', 'w') as of:
                for line in dataset:
                    of.write(line)
            with open(dir + '/' + 'vocab.source', 'w') as vs:
                with open(dir + '/' + 'vocab.target', 'w') as vt:
                    for elem in vocab:
                        vs.write(elem + '\n')
                        vt.write(elem + '\n')
            with open(dir + '/' + 'vocab.feats', 'w') as vf:
                for elem in feat_vocab:
                    vf.write(elem + '\n')


    def main(self):
        stems = list(self.gen_stems())[0:4000]
        random.shuffle(stems)
        a_class = stems[0:1000]
        b_class = stems[1000:2000]
        c_class = stems[2000:3000]
        d_class = stems[3000:4000]
        get = lambda x, y, z: self.gen_paradigm(self.prefixes,
                                 z,
                                 x,
                                 self.affixes,
                                 y)

        # self.gen_paradigm(self.prefixes, z, x, self.affixes, y)
        # pdb.set_trace()

        a_hiers = sorted(set().union(*[get(x, 'A', self.hier_classes) for x in a_class]))
        b_hiers = sorted(set().union(*[get(x, 'B', self.hier_classes) for x in b_class]))
        c_hiers = sorted(set().union(*[get(x, 'C', self.hier_classes) for x in c_class]))
        d_hiers = sorted(set().union(*[get(x, 'D', self.hier_classes) for x in d_class]))

        a_cross = sorted(set().union(*[get(x, 'A', self.cross_classes) for x in a_class]))
        b_cross = sorted(set().union(*[get(x, 'B', self.cross_classes) for x in b_class]))
        c_cross = sorted(set().union(*[get(x, 'C', self.cross_classes) for x in c_class]))
        d_cross = sorted(set().union(*[get(x, 'D', self.cross_classes) for x in d_class]))

        a_grid = sorted(set().union(*[get(x, 'A', self.grid_classes) for x in a_class]))
        b_grid = sorted(set().union(*[get(x, 'B', self.grid_classes) for x in b_class]))
        c_grid = sorted(set().union(*[get(x, 'C', self.grid_classes) for x in c_class]))
        d_grid = sorted(set().union(*[get(x, 'D', self.grid_classes) for x in d_class]))

        hiers = a_hiers + b_hiers + c_hiers + d_hiers
        cross = a_cross + b_cross + c_cross + d_cross
        grids = a_grid + b_grid + c_grid + d_grid

        random.shuffle(hiers)
        random.shuffle(cross)
        random.shuffle(grids)

        # a_hiers = [get(x, 'A', self.hier_classes) for x in a_class]
        # b_hiers = [get(x, 'B', self.hier_classes) for x in b_class]
        # c_hiers = [get(x, 'C', self.hier_classes) for x in c_class]
        # d_hiers = [get(x, 'D', self.hier_classes) for x in d_class]
        #
        # a_cross = [get(x, 'A', self.cross_classes) for x in a_class]
        # b_cross = [get(x, 'B', self.cross_classes) for x in b_class]
        # c_cross = [get(x, 'C', self.cross_classes) for x in c_class]
        # d_cross = [get(x, 'D', self.cross_classes) for x in d_class]
        #
        # a_grid = [get(x, 'A', self.grid_classes) for x in a_class]
        # b_grid = [get(x, 'B', self.grid_classes) for x in b_class]
        # c_grid = [get(x, 'C', self.grid_classes) for x in c_class]
        # d_grid = [get(x, 'D', self.grid_classes) for x in d_class]

        # hiers = a_hiers + b_hiers + c_hiers + d_hiers
        # cross = a_cross + b_cross + c_cross + d_cross
        # grids = a_grid + b_grid + c_grid + d_grid

        # random.shuffle(hiers)
        # random.shuffle(cross)
        # random.shuffle(grids)

        self.writeout(hiers, 'hier')

        self.writeout(cross, 'cross')

        self.writeout(grids, 'grid')

    def gen(self, struct, affixes, title):
        classes = list(struct.keys())
        thousands = len(classes) * 1000
        stems = list(self.gen_stems())[0:thousands]
        random.shuffle(stems)
        class_stem = []
        for i, infl_class in enumerate(classes):
            start = i * 1000
            end = (i + 1) * 1000
            class_stem.append(stems[start:end])

        prefixes = self.gen_pref(struct[random.choice(classes)])

        # z = classes
        # y = which class
        # x = stem
        get = lambda x, y, z: self.gen_paradigm(prefixes,
                                 z,
                                 x,
                                 affixes,
                                 y)

        dataset = []

        for i, infl_class in enumerate(classes):
            try:
                class_members = sorted(set().union(*[get(x, infl_class, struct) for x in tqdm.tqdm(class_stem[i])]))
            except:
                print("sorted(set().union(*[get(x, infl_class, struct) for x in class_stem[i]]))")
                print("didn't work")
                pdb.set_trace()
            dataset += class_members

        # a_hiers = sorted(set().union(*[get(x, 'A', self.hier_classes) for x in a_class]))

        # pdb.set_trace()

        random.shuffle(dataset)

        self.writeout(dataset, title)

    def gen_hier(self, struct, affixes, title):
        # normally these are inflection classes
        # here we only have cells
        cells = list(struct.keys())
        # thousands = len(classes) * 1000
        class_stem = list(self.gen_stems())[0:12000]
        random.shuffle(class_stem)

        prefixes = {}
        for cell in cells:
            cell_prefix = cell.split(' ')
            cell_prefix = ["OUT=" + x for x in cell_prefix]
            cell_prefix = ' '.join(cell_prefix)
            prefixes[cell] = cell_prefix

        dataset = []

        for stem in tqdm.tqdm(class_stem):
            in_form = "IN=sg IN=nom\t" + ' '.join(stem + affixes[struct['sg nom']])
            for cell in struct:
                suffix = affixes[struct[cell]]
                feats = prefixes[cell]
                entry = feats + ' ' + in_form + '\t' + ' '.join(stem + suffix) + '\n'
                dataset.append(entry)

        # a_hiers = sorted(set().union(*[get(x, 'A', self.hier_classes) for x in a_class]))

        random.shuffle(dataset)

        self.hier_writeout(dataset, title)

if __name__ == '__main__':
    g = GenLangs()
    # g.main()
    # g.gen(g.faux_latin_verbs, g.affxes, 'faux_latin')
    # g.gen(g.faux_russian_nouns, g.affixes, 'faux_russian')
    # g.gen_hier(g.faux_latin, g.hier_affixes, 'faux-num-case')
    # g.gen_hier(g.faux_latin, g.hier_invert_affixes, 'faux-case-num')
    # g.gen_hier(g.faux_latin, g.hier_invert_affixes_full, 'faux-case-num-full')
    g.gen_hier(g.faux_latin, g.case_num_affixes, 'faux-case-num')
    g.gen_hier(g.faux_latin, g.num_case_affixes, 'faux-num-case')
    g.gen_hier(g.faux_latin, g.case_num_1_case_affixes, 'faux-case-num-1-case')
    g.gen_hier(g.faux_latin, g.case_num_1_num_affixes, 'faux-case-num-1-num')
    g.gen_hier(g.faux_latin, g.case_case_num_2_affixes, 'faux-case-case-num-2')
    g.gen_hier(g.faux_latin, g.num_case_num_2_affixes, 'faux-num-case-num-2')