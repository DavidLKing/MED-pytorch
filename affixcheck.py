# from __future__ import print_function
# from __future__ import division
import sys
import difflib
import pickle as pkl
# import cPickle as pkl
# from getwrong import diffasstring
# from getwrong import spanningrule

class affixes:
    def __init__(self):
        pass

    def start(self, goldfile, wrongfile):
        self.goldlines = open(goldfile, 'r').readlines()
        self.wronglines = open(wrongfile, 'r').readlines()
        self.noun_affixes = set()
        self.verb_affixes = set()
        self.adj_affixes = set()
        print("goldlines:", len(self.goldlines))
        print("wrong lines:", len(self.wronglines))

    def diffasstring(self, word1, word2):
        editOP = []
        word1 = word1.strip()
        word2 = word2.strip()
        edits = difflib.ndiff(word1, word2)
        for e in edits:
            editOP.append(e)
        # print("Wordlevel Rule", wordlevel(editOP))
        wordrule = self.wordlevel(editOP)
        spanrule, affixes = self.spanningrule(editOP)
        # print("Span Rule", spanrule)
        # print("Affixes in question", affixes)
        return ''.join(editOP), wordrule, spanrule, affixes


    def array_diff(self, list1, list2):
        editOP = []
        edits = difflib.ndiff(list1, list2)
        for e in edits:
            editOP.append(e)
        # print("Wordlevel Rule", wordlevel(editOP))
        # wordrule = self.wordlevel(editOP)
        # spanrule, affixes = self.spanningrule(editOP)
        # print("Span Rule", spanrule)
        # print("Affixes in question", affixes)
        return editOP
        """
        We really only care about lem > gold misses
                                and lem > guess errors
        for edit rules:
            word level rules
            spanning word level rules
            affix rules
        """

    def wordlevel(self, editops):
        """
        Scan replace all non+ or - beginning letters with _
        remove spaces for legibility
        """
        rule = ''
        for edit in editops:
            # print("edit", edit)
            if edit[0] in ['+', '-']:
                rule += edit.replace(' ', '')
            else:
                rule += '_'
        return rule

    def spanningrule(self, editops):
        """
        Find all non- and non+ and remove them
        They delimit the segments
        """
        affixes = []
        rules = []
        affix = ''
        count = 0
        for edit in editops:
            # print("edit", edit)
            if edit[0] in ['+', '-']:
                affix += edit.replace(' ', '')
            elif edit[0] not in ['+', '-'] and affix != '':
                # the affix for spanning rules
                rules.append(affix)
                # our span delimiter
                rules.append('_')
                # segment rules
                # for legibility -- adding _ at the end for prefix,
                # and on both sides for infixes
                if count == 0:
                    affix = affix + '_'
                else:
                    affix = '_' + affix + '_'
                affixes.append(affix)
                # clear our affix
                affix = ''
            elif edit[0] not in ['+', '-'] and affix == '' and rules == []:
                rules.append('_')
            # using the count to determine if we're on the prefix or not
            # 0 = prefix
            count += 1
        # Get the final one:
        if affix != '':
            rules.append(affix)
            # must be a suffix
            affix = '_' + affix
            affixes.append(affix)
        return ''.join(rules), affixes

    def buildgold(self, goldlines):
        print("building gold affixes seen in training")
        for line in goldlines:
            line = line.strip().split()
            lemma = line[0]
            feats = line[1]
            gold = line[2]
            pos = feats.split(',')[0]
            # print(pos)
            editOP, word, span, affixes = a.diffasstring(lemma, gold)
            # print(affixes)
            for affix in affixes:
                if pos == 'pos=N':
                    self.noun_affixes.add(affix)
                elif pos == 'pos=V':
                    self.verb_affixes.add(affix)
                elif pos == 'pos=ADJ':
                    self.adj_affixes.add(affix)
        print("adj affixes", len(self.adj_affixes))
        print("\t adj aff in nouns", len(self.adj_affixes.intersection(self.noun_affixes)))
        print("\t adj aff in verbs", len(self.adj_affixes.intersection(self.verb_affixes)))
        print("verb affixes", len(self.verb_affixes))
        print("\t verb aff in nouns", len(self.verb_affixes.intersection(self.noun_affixes)))
        print("\t verb aff in adj", len(self.verb_affixes.intersection(self.adj_affixes)))
        print("noun affixes", len(self.noun_affixes))
        print("\t noun aff in adj", len(self.noun_affixes.intersection(self.adj_affixes)))
        print("\t noun aff in verb", len(self.noun_affixes.intersection(self.verb_affixes)))

    def find_pos(self, report_list):
        for annotation in report_list:
            if annotation[0:4] == 'pos=':
                return annotation

    def add_to_dict(self, value, pos, dictionary):
        if value not in dictionary[pos]:
            dictionary[pos][value] = 0
        dictionary[pos][value] += 1

    def builderror(self, wronglines):
        wrong_adj = []
        wrong_verb = []
        wrong_noun = []
        # TODO write out lines with annotation about wrong pos
        # TODO when it's wrong, when it's right, and when there might be syntretism
        newreport = open('newreport.csv', 'w')
        # nouns
        n_in_adj = 0
        n_in_adj_sync = 0
        n_in_verb = 0
        n_in_verb_sync = 0
        n_okay = 0
        n_noise = 0
        # adjectives
        adj_in_verb = 0
        adj_in_verb_sync = 0
        adj_in_n = 0
        adj_in_n_sync = 0
        adj_okay = 0
        adj_noise = 0
        # verbs
        verb_in_adj = 0
        verb_in_adj_sync = 0
        verb_in_n = 0
        verb_in_n_sync = 0
        verb_okay = 0
        verb_noise = 0
        # totals
        verbs = 0
        nouns = 0
        adj = 0
        # affix error counts
        affix_counts = {'pos=V': {}, 'pos=N': {}, 'pos=ADJ': {}}
        for line in wronglines:
            if wronglines.index(line) != 0:
                # newreport.write(line)
            # else:
                # print(line)
                line = line.strip().split(',')
                # print(line)
                lemma = line[0]
                gold = line[1]
                guess = line[2]
                # pos = line[16]
                pos = self.find_pos(line)
                # print(lemma, pos)
                editOP, word, span, affixes = a.diffasstring(lemma, guess)
                len_of_line = len(line)
                # print('line', line)
                # print("affixes", affixes)
                for affix in affixes:
                    # print('affix', affix)
                    if pos == 'pos=N':
                        # print('we got to pos=N')
                        nouns += 1
                        if affix in self.verb_affixes.difference(self.noun_affixes):
                            n_in_verb += 1
                            newaffix = affix + '-verb'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.adj_affixes.difference(self.noun_affixes):
                            n_in_adj += 1
                            newaffix = affix + '-adj'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.verb_affixes.intersection(self.noun_affixes):
                            n_in_verb_sync += 1
                            newaffix = affix + "-verb-sync"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.adj_affixes.intersection(self.noun_affixes):
                            n_in_adj_sync += 1
                            newaffix = affix + "-adj-sync"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.noun_affixes:
                            n_okay += 1
                            newaffix = affix + "-okay"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        else:
                            n_noise += 1
                            newaffix = affix + "-noise"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                    elif pos == 'pos=V':
                        verbs += 1
                        if affix in self.noun_affixes.difference(self.verb_affixes):
                            verb_in_n += 1
                            newaffix = affix + "-noun"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.adj_affixes.difference(self.verb_affixes):
                            verb_in_adj += 1
                            newaffix = affix + "-adj"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.noun_affixes.intersection(self.verb_affixes):
                            verb_in_n_sync += 1
                            newaffix = affix + "-noun-sync"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.adj_affixes.intersection(self.verb_affixes):
                            verb_in_adj_sync += 1
                            newaffix = affix + "-adj-sync"
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.verb_affixes:
                            verb_okay += 1
                            newaffix = affix + '-okay'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        else:
                            verb_noise += 1
                            newaffix = affix + '-noise'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                    elif pos == 'pos=ADJ':
                        adj += 1
                        if affix in self.verb_affixes.difference(self.adj_affixes):
                            adj_in_verb += 1
                            newaffix = affix + '-verb'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.noun_affixes.difference(self.adj_affixes):
                            adj_in_n += 1
                            newaffix = affix + '-adj'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.verb_affixes.intersection(self.adj_affixes):
                            adj_in_verb_sync += 1
                            newaffix = affix + '-verb-sync'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.noun_affixes.intersection(self.adj_affixes):
                            adj_in_n_sync += 1
                            newaffix = affix + '-adj-sync'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        elif affix in self.adj_affixes:
                            adj_okay += 1
                            newaffix = affix + '-okay'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                        else:
                            adj_noise += 1
                            newaffix = affix + '-noise'
                            line.append(newaffix)
                            self.add_to_dict(newaffix, pos, affix_counts)
                    # check to make sure we didn't miss anything or add too much
                    # print("len line", len(line))
                    # print("prev len", len_of_line)
                    # print("len affixes", len(affixes))
                assert(len(line) == len_of_line + len(affixes))
            newreport.write(','.join(line) + '\n')
        # print('affix count', affix_counts)
        # Nouns
        print("Nouns")
        print('bad noun affixes in adjective set', n_in_adj, n_in_adj / nouns)
        print('bad noun affixes in verb set', n_in_verb, n_in_verb / nouns)
        print('bad affixes both in nouns and adjectives:', n_in_adj_sync, n_in_adj_sync / nouns)
        print('bad affixes both in nouns and verbs:', n_in_verb_sync, n_in_verb_sync / nouns)
        print('bad noun affixes only in nouns', n_okay, n_okay / nouns)
        print('bad noun affixes never seen in training', n_noise, n_noise / nouns)
        print('noun check:', (n_in_adj + n_in_verb + n_noise + n_in_verb_sync + n_in_adj_sync + n_okay) / nouns)
        # Verbs
        print("Verbs")
        print('bad verb affixes in adjective set', verb_in_adj, verb_in_adj / verbs)
        print('bad verb affixes in noun set', verb_in_n, verb_in_n / verbs)
        print('bad affixes in both verbs and adjectives:', verb_in_adj_sync, verb_in_adj_sync / verbs)
        print('bad affixes in both verbs and nouns', verb_in_n_sync, verb_in_n_sync / verbs)
        print('bad verb affixes only in verbs', verb_okay, verb_okay / verbs)
        print('bad verb affixes never seen in training', verb_noise, verb_noise / verbs)
        print('verb check:', (verb_in_n + verb_in_adj + verb_noise + verb_in_adj_sync + verb_in_n_sync + verb_okay) / verbs)
        # Adjectives
        print("Adjectives")
        print('bad adjective affixes in noun set', adj_in_n, adj_in_n / adj)
        print('bad adjective affixes in verb set', adj_in_verb, adj_in_verb / adj)
        print('bad affixes in both adjectives and nouns:', adj_in_n_sync, adj_in_n_sync / adj)
        print('bad affixes in both adjectives and verbs:', adj_in_verb_sync, adj_in_verb_sync / adj)
        print('bad adjective affixes in adjectives', adj_okay, adj_okay / adj)
        print('bad adjective affixes never seen in training', adj_noise, adj_noise / adj)
        print('adjective check', (adj_noise + adj_in_verb + adj_okay + adj_in_n + adj_in_verb_sync + adj_in_n_sync) / adj)
        # Distributions
        print('Errors with affixes')
        for pos in affix_counts:
            print(pos)
            pos_sorted = sorted(affix_counts[pos], key=affix_counts[pos].get, reverse=True)
            for val in pos_sorted:
                print('\t', val, affix_counts[pos][val])


if __name__ == '__main__':
    a = affixes()
    a.start(sys.argv[1], sys.argv[2])
    a.buildgold(a.goldlines)
    a.builderror(a.wronglines)
