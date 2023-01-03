import numpy as np
import pickle as pkl
import pdb
from sklearn.neighbors import NearestNeighbors
# %matplotlib inline
from artif_langs_gen import GenLangs

class KNN_Homog_Score:
  def __init__(self) -> None:
    self.lang = GenLangs()
    self.affix_reverse = {}
    for cell in self.lang.affixes:
        self.affix_reverse[
            self.lang.affixes[cell]
        ] = cell
    # TODO add IC labels
    # pdb.set_trace()

  def load_pickle(self, filename):

    # filename = sys.argv[1]
    # pdb.set_trace()
    model_output = pkl.load(open(filename, 'rb'))
    points = [model_output[x]['embed'] for x in model_output]

    affixes = [model_output[x]['tgt'][-3] for x in model_output]

    in1_affixes = [model_output[x]['src'][-3] for x in model_output]

    in2_affixes = [model_output[x]['src'][-15] for x in model_output]

    cells = [model_output[x]['feats'] for x in model_output]

    return points, affixes, in1_affixes, in2_affixes, cells

  def get_knns(self, matrix):
    # minkowski + p=2 is same as euclidean
    nn_model = NearestNeighbors(n_neighbors=1, 
                                radius=1.0, 
                                algorithm='auto', 
                                leaf_size=30, 
                                metric='minkowski', 
                                p=2, 
                                # metric_params=None, 
                                n_jobs=-1
                                )
    return nn_model.fit(matrix)

  def get_score(self, filename, manually_set_exp=None):

      if manually_set_exp:
          lang_prefix = manually_set_exp
      else:
          lang_prefix = filename.split('/')[-2]
      fn_to_paradigm = {
          'big_simple' : self.lang.simple_big_classes,
          'aab' : self.lang.aab_classes,
          'aba' : self.lang.aba_classes,
          'abb' : self.lang.abb_classes,
          'simple' : self.lang.simple_classes,
          'caha_aab' : self.lang.aab_classes,
          'caha_aba': self.lang.aba_classes,
          'caha_abb' : self.lang.abb_classes,
          'anderson_aab' : self.lang.aab_classes,
          'anderson_aba': self.lang.aba_classes,
          'anderson_abb' : self.lang.abb_classes,
          'contrary': self.lang.contrary_classes,
          'nometa_nested': self.lang.nometa_nested_classes,
      }

      if manually_set_exp:
          paradigm = fn_to_paradigm[lang_prefix]
      else:
          # This is dumb, but this is how we're getting the paradigm
          upto = lang_prefix.index('0') - 1
          para_type = lang_prefix[0:upto]
          paradigm = fn_to_paradigm[para_type]

      points, affixes, in1, in2, cells = self.load_pickle(filename)

      # Getting infl class
      # ic for ic in paradigm if affix in affixes
      classes = []
      class_plus_affix = []
      for aff, inp1, inp2 in zip(affixes, in1, in2):
          possible = ''
          if lang_prefix == 'nometa_nested':
              infl_class = len(sorted(set([aff, inp1, inp2])))
              possible = str(infl_class)
          else:
              for ic in paradigm:
                  if self.affix_reverse[aff + 'i'] in paradigm[ic]:
                      possible += ic
          classes.append(possible)
          class_plus_affix.append(possible + '-' + aff)
      # pdb.set_trace()

      matrix = np.asarray(points).squeeze(1)
      # labels = affixes

      # pdb.set_trace()

      knn_model = self.get_knns(matrix)

      scores = {}

      # knn_model.kneighbors()[1]
      for labels in [
                        ('affixes', affixes), 
                        ('cells', cells),
                        ('IC', classes),
                        ('ICa', class_plus_affix)
                       ]:
        knn_homog = []
        for point, label in zip(knn_model.kneighbors()[1], labels[1]):
          same = 0
          total = 0
          for neighbor in point:
            if labels[1][neighbor] == label:
              same += 1
            total += 1
          knn_homog.append(same / total)

        score = sum(knn_homog)/len(knn_homog)
        scores[labels[0]] = score

      # print("KNN homogeniety =", score)
      return scores

# fig = px.scatter(
#   # fig = px.scatter_3d(
#   data_frame=dataset,
#   x='x',
#   y='y',
#   color='label',
#   color_discrete_sequence=px.colors.qualitative.Dark24
# )

# fig.update_layout(title="KNN Homogeneity Score: {}".format(score))

# fig.show()
# fig.write_html('temp.html', auto_play=False)
