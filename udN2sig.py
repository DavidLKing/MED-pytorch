import pdb
from random import shuffle

unimorph_train = open('data/russian/train/data.txt', 'r')

ud_forms = {}
ud_nouns = open('ud_nouns.conllu', 'r')

translate = {
    'Case=Nom': 'OUT=NOM',
    'Case=Acc': 'OUT=ACC',
    'Case=Gen': 'OUT=GEN',
    'Case=Ins': 'OUT=INS',
    'Case=Loc': 'OUT=ESS',
    'Case=Dat': 'OUT=DAT',
    'Number=Plur': 'OUT=PL',
    'Number=Sing': 'OUT=SG'
}

uniforms = set()
for line in unimorph_train:
    line = line.strip().split('\t')
    uniforms.add(line[1])

rejects = ['Case=Par', 'Case=Voc']

def reject(feats, rejects):
    feats = feats.split("|")
    yn = False
    for f in feats:
        if f in rejects:
            yn = True
    return yn

def conf2uni(feats, trans_dict):
    keeps = []
    newfeats = 'OUT=N '
    feats = feats.split("|")
    for f in feats:
        if f.startswith('Animacy'):
            keeps.append(f)
        elif f.startswith('Gender'):
            keeps.append(f)
        elif f in trans_dict:
            newfeats += trans_dict[f]
            newfeats += ' '
    return keeps, newfeats

uniUDs = set()

seen_forms = 0
total = 0

for line in ud_nouns:
    line = line.strip().split('\t')
    lemma = line[2].lower()
    form = line[1].lower()
    feats = line[5]

    if not reject(feats, rejects):
        keeps, feats = conf2uni(feats, translate)
        if len(feats[0:-1].split(' ')) == 3 and len(keeps) == 2:
            for char in lemma:
                feats += char
                feats += ' '
            feats = feats[0:-1]
            form = ' '.join(form)
            if form not in uniforms:
                uniUDs.add(" ".join(keeps) + '\t' + feats + '\t' + form)
        else:
            print('feats', feats)
            print('keeps', keeps)

# print(len(uniUDs))
# print()

# pdb.set_trace()
# for line in sorted(uniUDs):
    # print(line)
