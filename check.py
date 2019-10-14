from __future__ import print_function
import sys
import argparse
import difflib
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
                    type=str, 
                    required=True,                     
                    default=None,
                    help='Input file')
parser.add_argument('-o', '--output', 
                    type=str, 
                    required=True,                     
                    default=None,
                    help='Output file')
parser.add_argument('--wrong', action='store_true',
                    default=None,
                    help='Find errors (and not correct items)')
parser.add_argument('--correct', action='store_true',
                    default=None,
                    help='Find correct items (and not errors)')
parser.add_argument('--factored', action='store_true',
                    default=None,
                    help='Select if morphosytactic featres are factored (separated)')
parser.add_argument('--not-factored', action='store_true',
                    default=None,
                    help='Select if morphosytactic featres are not factored (separated)')

opt = parser.parse_args()



# For raw text files, set this to False, for LMU validation output, True
# val_out = False
val_out = True

def diffasstring(word1, word2):
    editOP = []
    edits = difflib.ndiff(word1, word2)
    for e in edits:
        editOP.append(e)
    # print("Wordlevel Rule", wordlevel(editOP))
    wordrule = wordlevel(editOP)
    spanrule, affixes = spanningrule(editOP)
    # print("Span Rule", spanrule)
    # print("Affixes in question", affixes)
    return ''.join(editOP), wordrule, spanrule, affixes

"""
We really only care about lem > gold misses
                        and lem > guess errors
for edit rules:
    word level rules
    spanning word level rules
    affix rules
"""

def wordlevel(editops):
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

def spanningrule(editops):
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

medoutput = open(opt.input, 'r').readlines()

assert(not (opt.correct and not opt.wrong)),"You cannot select both corrent and incorrect test items for analysis"

if opt.correct:
    right_wrong = lambda x, y: x == y
elif opt.wrong:
    right_wrong = lambda x, y: x != y

print("writing report to {}".format(opt.output))
report = open(opt.output, 'w')
header = ['lemma', 'gold', 'guess', 'correct edit', 'incorrect edit', 'feats']
report.write('\t'.join(header) + '\n')


#dictionaries for counting:
affixesGuessed = {}
affixesMissed = {}
spanrulesMissed = {}
wordrulesMissed = {}
spanrulesGuessed = {}
wordrulesGuessed = {}

for i in range(len(medoutput)):
    # if right_wrong(goldlines[i], guesslines[i]):
    line = medoutput[i].strip().split('\t')
    if len(line) >= 3:
        lemma = ''.join([x for x in line[0].split(' ') if len(x) == 1])
        feats = [x for x in line[0].split(' ') if len(x) > 1]
        gold = ''.join(line[1].split(' '))
        guess = ''.join(line[2].split(' '))
        # pdb.set_trace()
        # get the data
        lem2gues, l2gueword, l2guespan, l2gueaffixes = diffasstring(lemma, guess)
        lem2gold, l2gooword, l2goospan, l2gooaffixes = diffasstring(lemma, gold)
        # update the counts:
        if l2gueword not in wordrulesGuessed:
            wordrulesGuessed[l2gueword] = 0
        wordrulesGuessed[l2gueword] += 1
        if l2gooword not in wordrulesMissed:
            wordrulesMissed[l2gooword] = 0
        wordrulesMissed[l2gooword] += 1
        if l2guespan not in spanrulesGuessed:
            spanrulesGuessed[l2guespan] = 0
        spanrulesGuessed[l2guespan] += 1
        if l2goospan not in spanrulesMissed:
            spanrulesMissed[l2goospan] = 0
        spanrulesMissed[l2goospan] += 1
        for affix in l2gueaffixes:
            if affix not in affixesGuessed:
                affixesGuessed[affix] = 0
            affixesGuessed[affix] += 1
        for affix in l2gooaffixes:
            if affix not in affixesMissed:
                affixesMissed[affix] = 0
            affixesMissed[affix] += 1
        # gues2gol = diffasstring(guess, gold)
        if opt.wrong and gold != guess:
            outarray = [lemma, gold, guess, l2goospan, l2guespan]
            if opt.factored:
                for f in feats:
                    outarray.append(f)
            elif opt.not_factored:
                outarray.append(','.join(feats))
            outline = '\t'.join(outarray) + '\n'
            report.write(outline)

if opt.wrong == '--wrong':
    # start reporting
    guessedAff = open('guessedAffixes.csv', 'w')
    for affix in affixesGuessed:
        if affix[0] == '_' and affix[-1] == '_':
            affType = 'infix'
        elif affix[0] == '_' and affix[-1] != '_':
            affType = 'suffix'
        elif affix[0] != '_' and affix[-1] == '_':
            affType = 'prefix'
        else:
            print("Error with affix not being typed", affix)
        guessedAff.write(affix + ',' + str(affixesGuessed[affix]) + ',' + affType + '\n')
    missedAff = open('missedAffixes.csv', 'w')
    for affix in affixesMissed:
        if affix[0] == '_' and affix[-1] == '_':
            affType = 'infix'
        elif affix[0] == '_' and affix[-1] != '_':
            affType = 'suffix'
        elif affix[0] != '_' and affix[-1] == '_':
            affType = 'prefix'
        else:
            print("Error with affix not being typed", affix)
        missedAff.write(affix + ',' + str(affixesMissed[affix]) + ',' + affType + '\n')
    guessedSpan = open('guessedSpanRules.csv', 'w')
    for spanrule in spanrulesGuessed:
        guessedSpan.write(spanrule + ',' + str(spanrulesGuessed[spanrule]) + '\n')
    missedSpan = open('missedSpanRules.csv', 'w')
    for spanrule in spanrulesMissed:
        missedSpan.write(spanrule + ',' + str(spanrulesMissed[spanrule]) + '\n')
    guessedWord = open('guessedWordRules.csv', 'w')
    for wordrule in wordrulesGuessed:
        guessedWord.write(wordrule + ',' + str(wordrulesGuessed[wordrule]) + '\n')
    missedWord = open('missedWordRules.csv', 'w')
    for wordrule in wordrulesMissed:
        missedWord.write(wordrule + ',' + str(wordrulesMissed[wordrule]) + '\n')
