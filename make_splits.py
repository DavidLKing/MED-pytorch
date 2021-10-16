import sys
import random
import os
import pdb


class split:
    def __init__(self):
        pass

    def split_and_shuffle(self, lines):
        random.shuffle(lines)
        # 10 way split, 6 in train, 1 in dev, 3 in test
        interval = len(lines) // 10
        train = lines[0:interval*6]
        dev = lines[interval*6:interval*7]
        test = lines[interval*7:]
        print("Train:", len(train))
        print("Dev:", len(dev))
        print("Test:", len(test))
        return train, dev, test

    def writeout(self, lines, filename):
        with open(filename, 'w') as of:
            for line in lines:
                of.write(line)

if __name__ == '__main__':
    s = split()
    print("Usage: python make_splits.py data.txt")
    train, dev, test = s.split_and_shuffle(open(sys.argv[1], 'r').readlines())
    s.writeout(train, 'train/data.txt')
    s.writeout(dev, 'dev/data.txt')
    s.writeout(test, 'test/data.txt')
    os.system('cp vocab.source train/vocab.source')
    os.system('cp vocab.source dev/vocab.source')
    os.system('cp vocab.source test/vocab.source')
    os.system('cp vocab.target train/vocab.target')
    os.system('cp vocab.target dev/vocab.target')
    os.system('cp vocab.target test/vocab.target')
