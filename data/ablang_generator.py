import os
import sys
import random
import pdb

def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)

LANGS = {
    'ab' : ('a', 'b'),
    'paren' : ('(', ')')
}


class Generator:
    def __init__(self):
        pass

    def generate(self, lang, length):
        # Generate vocab file data
        try:
            vocab = sorted(set(lang))
        except:
            vocab = list(lang)
        # Generate data samples
        data = []
        for num in range(length):
            prefix = lang[0] * (num + 1)
            suffix = lang[1] * (num + 1)
            data.append(prefix + suffix)
        # test
        assert(len(data) == length)
        # Break off a dev and text group
        train_idx = int(0.7 * length)
        dev_idx = int(0.8 * length)
        test_idx = length
        random.shuffle(data)
        train = data[0:train_idx]
        dev = data[train_idx:dev_idx]
        test = data[dev_idx:]
        return vocab, train, dev, test


if __name__ == '__main__':
    for lang in LANGS:
        dir = 'ablang/' + lang + '/'
        make_dir(dir)
        g = Generator()
        vocab, train, dev, test = g.generate(LANGS[lang], 50)
        # make more training data
        # train *= 100
        make_dir(dir + 'train')
        make_dir(dir + 'dev')
        make_dir(dir + 'test')
        datasets = {'train': train,
                    'dev': dev,
                    'test': test}
        for set in datasets:
            working_dir = dir + set + '/'
            with open(working_dir + 'data.txt', 'w') as of:
                for datum in datasets[set]:
                    inputs = datum[0:int(len(datum) * 0.5)]
                    of.write(inputs + '\t' + datum + '\n')
            for vocabs in ['.source', '.target']:
                with open(working_dir + 'vocab' + vocabs, 'w') as of:
                    for entry in vocab:
                        of.write(entry + '\n')
