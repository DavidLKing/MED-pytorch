import sys
import pdb

no_vecs = open('UDdev.no.vecs.w.info.tsv', 'r').readlines()
vecs = open('UDdev.vecs.w.info.tsv', 'r').readlines()

class Class:
    def __init__(self):
        self.IA = {'б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ч', 'ц', 'ш', 'щ'}
        self.II = {'а', 'я'}
        self.IB = {'о', 'е'}
        # either IIIA if fem, IIIB if neut, IIIC if putz + masc, otherwise IA if masc
        self.soft = {'ь'}

class Errors:
    def __init__(self):
        self.classes = Class()

    def input_to_vars(self, line):
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

    def pull_errors(self, line):
        line = line.strip().split('\t')
        if line[1] == line[2]:
            error = False
        else:
            error = True
        lemma, features, form = self.input_to_vars([line[0], line[1]])
        an_gen = line[-1].split(' ')
        animacy = an_gen[0]
        gender = an_gen[1]
        pred = ''.join(line[2].split(' '))
        # masc fem neut
        if lemma[-1] in self.classes.IA:
            if gender == "Gender=Masc":
                class_type = "IA"
            elif gender == "Gender=Fem":
                class_type = "IIIA"
            else:
                class_type = "UNK_IA_NEUT?"
        elif lemma[-1] in self.classes.IB:
            if gender == "Gender=Neut":
                class_type = "IB"
            else:
                class_type = "UNK_IB_NON_NEUT?"
        elif lemma[-1] in self.classes.II:
            if gender in ["Gender=Fem", "Gender=Masc"]:
                class_type = "II"
            else:
                class_type = "UNK_II_NEUT"
        elif lemma[-1] in self.classes.soft:
            if gender == "Gender=Fem":
                class_type = "IIIA"
            elif gender == "Gender=Neut":
                class_type = "IIIB"
            elif gender == "Gender=Masc":
                if lemma != 'путь':
                    class_type = "IA"
                else:
                    class_type = "IIIC"
            else:
                class_type = "UNK_soft"
        else:
            class_type = "UNK_total"
        return class_type, error, gender, animacy, lemma, features, pred

    def modify_counts(self, counts, class_type, error, gender, animacy, lemma, features, pred):\
        # if len(features) == 3:
        if error:
            err_type = 'error'
        else:
            err_type = 'correct'
        if class_type not in counts[err_type]:
            counts[err_type][class_type] = {}
        # if gender not in counts[err_type][class_type]:
        #     counts[err_type][class_type][gender] = 0
        num = features[2]
        case = features[1]
        if case not in counts[err_type][class_type]:
            counts[err_type][class_type][case] = {}
        if num not in counts[err_type][class_type][case]:
            counts[err_type][class_type][case][num] = 0
        counts[err_type][class_type][case][num] += 1
        return counts

    def pretty_print(self, vec_counts, no_vec_counts):
        header = ['no_vec_class', 'case', 'num', 'no_vec_errs', 'no_vec_total', 'no_vec_rate', 'vec_errs', 'vec_total', 'vec_rate']
        print('\t'.join(header))
        for no_vec_class in no_vec_counts['error']:
            for case in no_vec_counts['error'][no_vec_class]:
                for num in no_vec_counts['error'][class_type][case]:
                    try:
                        no_vec_errs = no_vec_counts['error'][no_vec_class][case].get(num, 0)
                    except:
                        no_vec_errs = 0
                    try:
                        no_vec_corr = no_vec_counts['correct'][no_vec_class][case].get(num, 0)
                    except:
                        no_vec_corr = 0
                    no_vec_total = no_vec_errs + no_vec_corr
                    try:
                        no_vec_rate = no_vec_errs / no_vec_total
                    except:
                        no_vec_rate = 0.0
                    try:
                        vec_errs = vec_counts['error'][no_vec_class][case].get(num, 0)
                    except:
                        vec_errs = 0
                    try:
                        vec_corr = vec_counts['correct'][no_vec_class][case].get(num, 0)
                    except:
                        vec_corr = 0
                    vec_total = vec_errs + vec_corr
                    try:
                        vec_rate = vec_errs / vec_total
                    except:
                        vec_rate = 0.0
                    print('\t'.join([str(x) for x in [no_vec_class, case, num, no_vec_errs, no_vec_total, no_vec_rate, vec_errs, vec_total, vec_rate]]))



if __name__ == '__main__':
    no_vec_counts = {
        'error': {},
        'correct': {}
    }
    vec_counts = {
        'error': {},
        'correct': {}
    }
    e = Errors()
    for line1, line2 in zip(no_vecs, vecs):
        class_type, error, gender, animacy, lemma, features, pred = e.pull_errors(line1)
        no_vec_counts = e.modify_counts(no_vec_counts, class_type, error, gender, animacy, lemma, features, pred)
        class_type, error, gender, animacy, lemma, features, pred = e.pull_errors(line2)
        vec_counts = e.modify_counts(vec_counts, class_type, error, gender, animacy, lemma, features, pred)

    e.pretty_print(vec_counts, no_vec_counts)
