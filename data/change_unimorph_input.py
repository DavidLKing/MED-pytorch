#!/usr/bin/env python3
import random
import sys
import argparse
import logging
import math
import os
import pdb

class InputReorder:
    def __init__(self):
        pass


    def load_file(self, infile):
        paradigms = {}
        for line in open(infile, 'r'):
            line = line.strip().split('\t')
            assert(len(line)==2)
            inform = line[1]
            # inputs = line[1].split(' ')
            outputs = line[2]
            feats = line[0]
            # feats = ' '.join([x for x in inputs if len(x) > 1])
            # inform = ' '.join([x for x in inputs if len(x)==1])
            if inform not in paradigms:
                paradigms[inform] = {}
            if feats not in paradigms[inform]:
                paradigms[inform][feats] = set()
            paradigms[inform][feats].add(outputs)
        return paradigms

    def change_key(self, paradigms):
        new_paradigms = {}
        has_inf = 0
        no_inf = 0
        for inform in paradigms:
            if "OUT=V OUT=NFIN" in paradigms[inform].keys():
                for infinitive in paradigms[inform]["OUT=V OUT=NFIN"]:
                    new_paradigms[infinitive] = paradigms[inform]
            else:
                new_paradigms[inform] = paradigms[inform]
        return new_paradigms

    def writeout(self, paradigms, prefix):
        # I apparently decided on a 60/10/30 split at sometime
        all_lines = []
        for form in paradigms:
            for feats in paradigms[form]:
                for output in paradigms[form][feats]:
                    newline = feats + '\t' + form + '\t' + output + '\n'
                    all_lines.append(newline)
        # shuffle and split
        random.shuffle(all_lines)
        chunk = len(all_lines) // 10
        train = all_lines[0:6*chunk]
        dev = all_lines[6*chunk:7*chunk]
        test = all_lines[8*chunk:]
        for dataset, filename in zip([train, dev, test], ['train', 'dev', 'test']):
            with open(prefix + '/' + filename + '/data.txt', 'w') as outfile:
                for entry in dataset:
                    outfile.write(entry)
        print("Files written")

if __name__ == '__main__':
    # Command line arguments
    # print("Sample usage: ./prototype.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-file', help="MED formatted information", required=True)
    # requiredNamed = parser.add_argument_group('required named arguments')
    # requiredNamed.add_argument('-i', '--input', help='Input file name', required=True)
    args = parser.parse_args()
    # Logging
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(filename='prototype.log', level=logging.DEBUG)
    # logger.info("Options:\n{}".format(pprint.pformat(self.args)))

    r = InputReorder()
    data_paradigms = r.load_file(args.data_file)
    inf_paradigms = r.change_key(data_paradigms)
    prefix = '/'.join(args.data_file.split('/')[:-1])
    r.writeout(inf_paradigms, prefix)

