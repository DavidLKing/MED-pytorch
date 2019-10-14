import pdb
# coding: utf-8

# In[1]:


import difflib
import pandas as pd

from affixcheck import affixes


# In[2]:


a = affixes()


# In[3]:


def gen_paradigms(unis):
    paradigms = {}
    for line in unis:
        line = line.strip().split('\t')
        if len(line) > 1:
            assert(len(line) == 3)
            # TODO fix space bug
            # if lemma == 'геркулесоваякаша':
            # should be геркулесовая каша
            #     pdb.set_trace()
            lemma = line[0].replace(' ', '')
            word = line[1].replace(' ', '')
            features = line[2]
            if lemma not in paradigms:
                paradigms[lemma] = {}
            if features not in paradigms[lemma]:
                paradigms[lemma][features] = word
    return paradigms


# In[4]:


def get_errors(outputs):
    errors = []
    for line in outputs:
        line = line.strip().split('\t')
        if len(line) == 3:
            if line[1] != line[2]:
                errors.append(line)
    return errors


# In[5]:


unimorph = open('data/russian/uni/rus-fake-train.tsv', 'r')
train = open('data/russian/train/data.txt', 'r')
dev = open('data/russian/dev/data.txt', 'r')
vecs = open('russian-w-vecs.tsv', 'r')
novecs = open('russian-no-vecs.tsv', 'r')
paradigms = gen_paradigms(unimorph)
vec_errors = get_errors(vecs)
novec_errors = get_errors(novecs)


# In[6]:


def get_cite(form_list):
    input_forms = []
    for cite in form_list:
        cite_input = []
        cite = cite.split(' ')
        for elem in cite:
            if '=' not in elem:
                cite_input.append(elem)
        input_forms.append(''.join(cite_input))
    return input_forms


# In[7]:


def input_to_vars(line):
    lemma = []
    features = []
    word = []
    for char in line[0].split(' '):
        # TODO fix space bug
        # if lemma == 'геркулесоваякаша':
        # should be геркулесовая каша
        #     pdb.set_trace()
        if '=' not in char:
            # if char == '':
                # char = char.replace('', ' ')
            lemma.append(char)
        else:
            features.append(char)
    lemma = ''.join(lemma)
    for char in line[1].split(' '):
        # if char == '':
            # char = char.replace('', ' ')
        word.append(char)
    word = ''.join(word)
    return lemma, features, word


# In[8]:


def i_e_counts(unimorph, paradigms):
    i_count = 0
    e_count = 0
    i_imp = 0
    e_imp = 0
    i_per = 0
    e_per = 0
    missing = 0
    unimorph.seek(0)
    for line in unimorph:
        line = line.strip().split('\t')
        if len(line) > 1:
            assert(len(line) == 2)
            lemma, features, word = input_to_vars(line)
            if "OUT=V" in features:
                inform = 'V;PRS;2;SG'
                if lemma in paradigms:
                    if inform in paradigms[lemma]:
                        second_sing = paradigms[lemma][inform]
                        _, _, _, affixes = a.diffasstring(lemma, second_sing)
                        for affix in affixes:
                            if '+е' in affix or '+ё' in affix:
                                e_count += 1
                                if 'OUT=IPFV' in features:
                                    e_imp += 1
                                elif 'OUT=PFV' in features:
                                    e_per += 1
                            else:
                                i_count += 1
                                if 'OUT=IPFV' in features:
                                    i_imp += 1
                                elif 'OUT=PFV' in features:
                                    i_per += 1
                else:
                    missing += 1
    print("Missing citation forms", missing)
    print("i-conj + impf", i_imp)
    print("i-conj + perf", i_per)
    print("e-conj + impf", e_imp)
    print("e-conj + per", e_per)
    return i_count, e_count
                


# In[9]:


def class_error(conj_class, affixes):
    class_error = False
    for affix in affixes:
        if conj_class in ['е-conj', 'ё-conj']:
            if '-е' in affix or '-ё' in affix or '+и' in affix:
                class_error = True
        elif conj_class == 'и-conj':
            if '-и' in affix or '+ё' in affix:
                class_error = True
    return class_error


# In[10]:


def get_verb_class(inputs, golds, preds, paradigms, i_class_total, e_class_total):
    missing = 0
    total = 0
    error = 0
    class_errors = 0
    i_class_errors = 0
    e_class_errors = 0
    
    for cite, form, pred in zip(inputs, golds, preds):
        total += 1
        form = ''.join(form.split(' '))
        pred = ''.join(pred.split(' '))
        if cite in paradigms:
            inform = 'V;PRS;2;SG'
            if inform in paradigms[cite]:
                second_sing = paradigms[cite][inform]
                _, _, _, affixes = a.diffasstring(cite, second_sing)
                _, _, _, pred_affixes = a.diffasstring(cite, pred)
                _, _, _, form_affixes = a.diffasstring(cite, form)
                _, _, _, diff_affixes = a.diffasstring(form, pred)
                for affix in affixes:
                    if '+е' in affix:
                        conj_class = 'е-conj'
                        error_type = class_error(conj_class, diff_affixes)
                        e_class_errors += 1
                    elif '+ё' in affix:
                        conj_class = 'ё-conj'
                        error_type = class_error(conj_class, diff_affixes)
                        e_class_errors += 1
                    else:
                        conj_class = 'и-conj'
                        error_type = class_error(conj_class, diff_affixes)
                        i_class_errors += 1
                # print(cite, second_sing, conj_class, form, pred, error_type, diff_affixes )
                # pdb.set_trace()
                if error_type:
                    class_errors += 1
            else:
                missing += 1
        else:
            error += 1
    
    print("Missing", missing, "of", total)
    print("Total found", total - missing)
    print("Missing", error, "citation forms (errors)")
    print("Total class error detected:", class_errors, "or", class_errors / (total - missing))
    print("i_conj errors", i_class_errors, i_class_errors / (total - missing))
    print("i_conj error rate", i_class_errors / i_class_total)
    print("i_conj total", i_class_total)
    print("e_conj errors", e_class_errors, e_class_errors / (total - missing))
    print("e_conj error rate", e_class_errors / e_class_total)
    print("e_conj total", e_class_total)


# _Verbs!_

# In[11]:


train_i, train_e = i_e_counts(train, paradigms)
print("Total i_conj in train:", train_i)
print("Total e_conj in train:", train_e)


# ```
# grep OUT=V data/russian/train/data.txt | wc -l
# 3690
# ```

# In[12]:


i_class_total, e_class_total = i_e_counts(dev, paradigms)


# In[13]:


v_errors = pd.DataFrame(vec_errors, columns=['input', 'gold', 'pred'])
verbs = v_errors[v_errors['input'].str.match('OUT=V')]
inputs = get_cite(list(verbs['input']))
golds = list(verbs['gold'])
preds = list(verbs['pred'])
get_verb_class(inputs, golds, preds, paradigms, i_class_total, e_class_total)


# In[14]:


nv_errors = pd.DataFrame(novec_errors, columns=['input', 'gold', 'pred'])
verbs = nv_errors[nv_errors['input'].str.match('OUT=V')]
inputs = get_cite(list(verbs['input']))
golds = list(verbs['gold'])
preds = list(verbs['pred'])
get_verb_class(inputs, golds, preds, paradigms, i_class_total, e_class_total)


# _Nouns!_

# In[15]:


class Gender:
    def __init__(self, masc, fem, neut, masc_fem):
        self.masc = masc
        self.fem = fem
        self.neut = neut
        self.masc_fem = masc_fem
        
masc = {'б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ч', 'ц', 'ш','щ'}
fem = {'а', 'я'}
neut = {'о', 'е'}
masc_fem = {'ь'}
classes = Gender(masc, fem, neut, masc_fem)


# In[68]:


def anim_error(lemma, form, single, paradigms, error_total, error_count, found_count):
#     error = 'not found'
    
    if single:
        inform_acc = 'N;ACC;SG'
        inform_nom = 'N;NOM;SG'
        inform_gen = 'N;GEN;SG'
    else:
        inform_acc = 'N;ACC;PL'
        inform_nom = 'N;NOM;PL'
        inform_gen = 'N;GEN;PL'
    
    if lemma in paradigms:
        if inform_acc in paradigms[lemma] and            inform_gen in paradigms[lemma] and            inform_nom in paradigms[lemma]:
            # print(form, paradigms[lemma][inform_acc])
            # assert(paradigms[lemma][inform_acc] != form)
            if form[-2:] != paradigms[lemma][inform_acc][-2:]:
                if form[-2:] == paradigms[lemma][inform_gen][-2:] or                    form[-2:] == paradigms[lemma][inform_nom][-2:]:
                    # if (paradigms[lemma][inform_gen] == paradigms[lemma][inform_acc]) or \
                    #    (paradigms[lemma][inform_nom] == paradigms[lemma][inform_acc]):
                    # anim_error = 'anim'
                    error_count += 1
                    found_count += 1
                    error_total += 1
            elif form[-2:] != paradigms[lemma][inform_gen][-2:] and                  form[-2:] != paradigms[lemma][inform_nom][-2:]:
                    # anim_error = 'other'
                    found_count += 1
                    error_total += 1
            
    return error_total, error_count, found_count


# In[69]:


def class_count(dev, classes):
    masc_acc_sg_total = 0
    fem_acc_sg_total = 0
    neut_acc_sg_total = 0
    masc_fem_acc_sg_total = 0
    masc_acc_pl_total = 0
    fem_acc_pl_total = 0
    neut_acc_pl_total = 0
    masc_fem_acc_pl_total = 0
    
    dev.seek(0)
    
    for line in dev:
        line = line.split('\t')
        if len(line) == 2:
            lemma, features, word = input_to_vars(line)
            if 'OUT=N' in features:
                if 'OUT=ACC' in features:
                    if 'OUT=SG' in features and lemma[-1] in classes.masc:
                        masc_acc_sg_total += 1
                    elif 'OUT=SG' in features and lemma[-1] in classes.fem:
                        fem_acc_sg_total += 1
                    elif 'OUT=SG' in features and lemma[-1] in classes.neut:
                        neut_acc_sg_total += 1
                    elif 'OUT=SG' in features and lemma[-1] in classes.masc_fem:
                        masc_fem_acc_sg_total += 1
                    elif 'OUT=PL' in features and lemma[-1] in classes.masc:
                        masc_acc_pl_total += 1
                    elif 'OUT=PL' in features and lemma[-1] in classes.fem:
                        fem_acc_pl_total += 1
                    elif 'OUT=PL' in features and lemma[-1] in classes.neut:
                        neut_acc_pl_total += 1
                    elif 'OUT=PL' in features and lemma[-1] in classes.masc_fem:
                        masc_fem_acc_pl_total += 1
    return masc_acc_sg_total, fem_acc_sg_total, neut_acc_sg_total, masc_fem_acc_sg_total, masc_acc_pl_total, fem_acc_pl_total, neut_acc_pl_total, masc_fem_acc_pl_total
    


# In[70]:


class_count(dev, classes)


# In[71]:


def error_count(errors, train, dev, classes, paradigms):
    total = 0
    found = 0
    
    masc_acc_sg_total = 0
    fem_acc_sg_total = 0
    neut_acc_sg_total = 0
    masc_fem_acc_sg_total = 0
    masc_acc_pl_total = 0
    fem_acc_pl_total = 0
    neut_acc_pl_total = 0
    masc_fem_acc_pl_total = 0

    class_counts = class_count(dev, classes)
    dev_masc_acc_sg_total = class_counts[0]
    dev_fem_acc_sg_total = class_counts[1]
    dev_neut_acc_sg_total = class_counts[2]
    dev_masc_fem_acc_sg_total = class_counts[3]
    dev_masc_acc_pl_total = class_counts[4]
    dev_fem_acc_pl_total = class_counts[5]
    dev_neut_acc_pl_total = class_counts[6]
    dev_masc_fem_acc_pl_total = class_counts[7]
    
    masc_acc_sg_anim_error = 0
    fem_acc_sg_anim_error = 0
    neut_acc_sg_anim_error = 0
    masc_fem_acc_sg_anim_error = 0
    masc_acc_pl_anim_error = 0
    fem_acc_pl_anim_error = 0
    neut_acc_pl_anim_error = 0
    masc_fem_acc_pl_anim_error = 0
    
    for inputs, gold, pred in zip(errors['input'], errors['gold'], errors['pred']):
        lemma, features, form = input_to_vars([inputs, pred])
        if "OUT=N" in features and "OUT=ACC" in features:
            total += 1
            # print(inputs, gold, pred)
            if "OUT=SG" in features:
                if lemma[-1] in classes.masc:
                    masc_acc_sg_total, masc_acc_sg_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  True, 
                                                                                  paradigms, 
                                                                                  masc_acc_sg_total, 
                                                                                  masc_acc_sg_anim_error, 
                                                                                  found)
                elif lemma[-1] in classes.fem:
                    fem_acc_sg_total, fem_acc_sg_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  True, 
                                                                                  paradigms, 
                                                                                  fem_acc_sg_total, 
                                                                                  fem_acc_sg_anim_error, 
                                                                                  found)
                elif lemma[-1] in classes.neut:
                    neut_acc_sg_total, neut_acc_sg_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  True, 
                                                                                  paradigms, 
                                                                                  neut_acc_sg_total, 
                                                                                  neut_acc_sg_anim_error, 
                                                                                  found)
                elif lemma[-1] in classes.masc_fem:
                    masc_fem_acc_sg_total, masc_fem_acc_sg_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  True, 
                                                                                  paradigms, 
                                                                                  masc_fem_acc_sg_total, 
                                                                                  masc_fem_acc_sg_anim_error, 
                                                                                  found)
            elif "OUT=PL" in features:
                if lemma[-1] in classes.masc:
                    masc_acc_pl_total, masc_acc_pl_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  False, 
                                                                                  paradigms, 
                                                                                  masc_acc_pl_total, 
                                                                                  masc_acc_pl_anim_error, 
                                                                                  found)
                elif lemma[-1] in classes.fem:
                    fem_acc_pl_total, fem_acc_pl_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  False, 
                                                                                  paradigms, 
                                                                                  fem_acc_pl_total, 
                                                                                  fem_acc_pl_anim_error, 
                                                                                  found)
                elif lemma[-1] in classes.neut:
                    neut_acc_pl_total, neut_acc_pl_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  False, 
                                                                                  paradigms, 
                                                                                  neut_acc_pl_total, 
                                                                                  neut_acc_pl_anim_error, 
                                                                                  found)
                elif lemma[-1] in classes.masc_fem:
                    masc_fem_acc_pl_total, masc_fem_acc_pl_anim_error, found = anim_error(lemma, 
                                                                                  form, 
                                                                                  True, 
                                                                                  paradigms, 
                                                                                  masc_fem_acc_pl_total, 
                                                                                  masc_fem_acc_pl_anim_error, 
                                                                                  found)
    
    
    
    
    if masc_acc_sg_total > 0:
        print("masc_acc_sg_anim_error", masc_acc_sg_anim_error, 'of', dev_masc_acc_sg_total)#masc_acc_sg_anim_error / masc_acc_sg_total)
        print("masc_acc_sg_total", masc_acc_sg_total)
        print("masc_acc_sg_anim_error rate on dev", masc_acc_sg_anim_error / dev_masc_acc_sg_total)
    
    if fem_acc_sg_total > 0:
        print("fem_acc_sg_anim_error", fem_acc_sg_anim_error, 'of', dev_fem_acc_sg_total)#fem_acc_sg_anim_error / fem_acc_sg_total)
        print("fem_acc_sg_total", fem_acc_sg_total)
        print("fem_acc_sg_anim_error rate on dev", fem_acc_sg_anim_error / dev_fem_acc_sg_total)
    
    if neut_acc_sg_total > 0:
        print("neut_acc_sg_anim_error", neut_acc_sg_anim_error, 'of', dev_neut_acc_sg_total)#neut_acc_sg_anim_error /neut_acc_sg_total )
        print("neut_acc_sg_total", neut_acc_sg_total)
        print("neut_acc_sg_anim_error rate on dev", neut_acc_sg_anim_error / dev_neut_acc_sg_total)
        
    if masc_fem_acc_sg_total > 0:
        print("masc_fem_acc_sg_anim_error", masc_fem_acc_sg_anim_error, 'of', dev_masc_fem_acc_sg_total)#neut_acc_sg_anim_error /neut_acc_sg_total )
        print("masc_fem_acc_sg_total", masc_fem_acc_sg_total)
        print("masc_fem_acc_sg_anim_error rate on dev", masc_fem_acc_sg_anim_error / dev_masc_fem_acc_sg_total)

    if masc_acc_pl_total > 0:
        print("masc_acc_pl_anim_error", masc_acc_pl_anim_error, 'of', dev_masc_acc_pl_total)#masc_acc_pl_anim_error / masc_acc_pl_total)
        print("masc_acc_pl_total", masc_acc_pl_total)
        print("masc_acc_pl_anim_error rate on dev", masc_acc_pl_anim_error / dev_masc_acc_pl_total)

    if fem_acc_pl_total > 0:
        print("fem_acc_pl_anim_error", fem_acc_pl_anim_error, 'of', dev_fem_acc_pl_total)#fem_acc_pl_anim_error / fem_acc_pl_total)
        print("fem_acc_pl_total", fem_acc_pl_total)
        print("fem_acc_pl_anim_error rate on dev", fem_acc_pl_anim_error / dev_fem_acc_pl_total)

    if neut_acc_pl_total > 0:
        print("neut_acc_pl_anim_error", neut_acc_pl_anim_error, 'of', dev_neut_acc_pl_total)#neut_acc_pl_anim_error / neut_acc_pl_total)
        print("neut_acc_pl_total", neut_acc_pl_total)
        print("neut_acc_pl_anim_error rate on dev", neut_acc_pl_anim_error / dev_neut_acc_pl_total)
        
    if masc_fem_acc_sg_total > 0:
        print("masc_fem_acc_pl_anim_error", masc_fem_acc_pl_anim_error, 'of', dev_masc_fem_acc_pl_total)#neut_acc_sg_anim_error /neut_acc_sg_total )
        print("masc_fem_acc_pl_total", masc_fem_acc_pl_total)
        print("masc_fem_acc_pl_anim_error rate on dev", masc_fem_acc_pl_anim_error / dev_masc_fem_acc_pl_total)
    
    print('found', found, 'paradigm entries for N;ACC;SG/PL of', total)
    print("sanity check", sum(list(class_counts)))


# ```
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "N;ACC;ANIM" data/russian/rus-fake-train.tsv | wc -l
# 352
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "N;ACC" data/russian/rus-fake-train.tsv | wc -l
# 33682
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "N;ACC;SG" data/russian/rus-fake-train.tsv | wc -l
# 15507
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "N;ACC;PL" data/russian/rus-fake-train.tsv | wc -l
# 12036
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "OUT=N OUT=ACC OUT=ANIM" data/russian/dev/data.txt | wc -l
# 23
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "OUT=N OUT=ACC" data/russian/dev/data.txt | wc -l
# 1514
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "OUT=N OUT=ACC OUT=SG" data/russian/dev/data.txt | wc -l
# 827
# (pytorch-seq2seq) david@Arjuna:~/bin/git/MED-pytorch$ grep "OUT=N OUT=ACC OUT=PL" data/russian/dev/data.txt | wc -l
# 647
# ```

# In[72]:


error_count(v_errors, train, dev, classes, paradigms)


# In[73]:


error_count(nv_errors, train, dev, classes, paradigms)

