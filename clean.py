import sys
import pdb

newlines = []

_file = sys.argv[1]

lang_prefix = _file[0:2]
# print(lang_prefix)

infile = open(_file, 'r')

def eng_cop(line):
    feats = line[5]
    if feats == "Abbr=Yes|Mood=Ind|Tense=Pres|VerbForm=Fin":
        return "ar"
    elif feats == "Mood=Imp|Person=3|Tense=Pres|VerbForm=Fin":
        return "be"
    elif feats == "Mood=Imp|VerbForm=Fin":
        return "be"
    elif feats == "Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin":
        return "were"
    elif feats == "Mood=Ind|Number=Plur|Tense=Pres|VerbForm=Fin":
        return "are"
    elif feats == "Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin":
        return "was"
    elif feats == "Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin":
        return "am"
    elif feats == "Mood=Ind|Number=Sing|Person=2|Tense=Past|VerbForm=Fin":
        return "were"
    elif feats == "Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin":
        return "are"
    elif feats == "Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin":
        return "was"
    elif feats == "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin":
        return "is"
    elif feats == "Mood=Ind|Tense=Past|VerbForm=Fin":
        return "were"
    elif feats == "Mood=Ind|Tense=Pres|VerbForm=Fin":
        return "are"
    elif feats == "Mood=Sub|Person=3|Tense=Pres|VerbForm=Fin":
        return "be"
    elif feats == "Mood=Sub|VerbForm=Fin":
        return "be"
    elif feats == "Number=Sing|Tense=Pres|VerbForm=Part":
        return "being"
    elif feats == "Person=3|VerbForm=Ger":
        return "being"
    elif feats == "Tense=Past|VerbForm=Part":
        return "been"
    elif feats == "Tense=Pres|VerbForm=Part":
        return "being"
    elif feats == "VerbForm=Ger":
        return "being"
    elif feats == "VerbForm=Inf":
        return "be"
    else:
        return "be"

for line in infile:
    line = line.strip().split('\t')
    if len(line) > 2:
        if line[3] in ['PROPN', 'PUNCT', 'NUM']:
            line[2] = line[1]
        if lang_prefix == 'en' and line[4] == 'VBG':
            if line[1] != 'be':
                if line[1].endswith('e'):
                    line[2] = line[1][:-1] + 'ing'
                else:
                    line[2] = line[1] + 'ing'
            else:
                line[2] = line[1] + 'ing'
        elif lang_prefix == "en" and line[1] == 'be':
            line[2] = eng_cop(line)

    newlines.append('\t'.join(line) + '\n')

infile.close()
outfile = open(sys.argv[1], 'w')
for line in newlines:
    outfile.write(line)

outfile.close()