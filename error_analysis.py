
# coding: utf-8

# In[1]:

import pdb
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
            lemma = line[0]
            word = line[1]
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


# In[65]:


unimorph = open('data/russian/rus-fake-train.tsv', 'r')
vecs = open('russian-w-vecs.tsv', 'r')
novecs = open('russian-no-vecs.tsv', 'r')
paradigms = gen_paradigms(unimorph)
vec_errors = get_errors(vecs)
novec_errors = get_errors(novecs)


# In[39]:


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


# In[79]:
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

def get_verb_class(inputs, golds, preds, paradigms):
    missing = 0
    total = 0
    error = 0
    class_errors = 0
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
                    elif '+ё' in affix:
                        conj_class = 'ё-conj'
                        error_type = class_error(conj_class, diff_affixes)
                    else:
                        conj_class = 'и-conj'
                        error_type = class_error(conj_class, diff_affixes)
                # print(cite, second_sing, conj_class, form, pred, error_type, diff_affixes )
                # pdb.set_trace()
                if error_type:
                    class_errors += 1
            else:
                missing += 1
        else:
            error += 1
    print("Missing", missing, "of", total)
    print("Missing", error, "citation forms (errors)")
    print("Total class error detected:", class_errors)

# In[78]:


v_errors = pd.DataFrame(vec_errors, columns=['input', 'gold', 'pred'])
verbs = v_errors[v_errors['input'].str.match('OUT=V')]
inputs = get_cite(list(verbs['input']))
golds = list(verbs['gold'])
preds = list(verbs['pred'])
get_verb_class(inputs, golds, preds, paradigms)

