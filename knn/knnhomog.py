from tqdm import tqdm
import os
import sys
import numpy as np
import random
import pickle as pkl
import pdb
import sklearn
import pandas as pd
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
# from cuml.manifold import TSNE
# from tsnecuda import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors,
                     metrics)
from sklearn.cluster import dbscan, OPTICS
from sklearn.metrics import normalized_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score

import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from knnscore import KNN_Homog_Score

knnhomog = KNN_Homog_Score()

# test = "/Users/david/Desktop/hypertune2/aba-0.65-150-testing/Finished epoch 30: Train Perplexity: 1.0002, Dev Perplexity: 1.0002, Accuracy: 8.1675.pkl"
# print(knnhomog.get_score(test))

prefix = sys.argv[1]

# exp_filename = prefix.split('/')[-2]
exp_filename = 'simple'
# pdb.set_trace()

datas = {
  'exp': [],
  'score': [],
  'label': []
}

for exp in tqdm(sorted(os.listdir(prefix), key=len)):
  # dumb mac .DS_Store thing
  if "/." not in prefix + "/" + exp and 'nested' not in exp:
    epochs = os.listdir(prefix + "/" + exp)
    # Only necessary for now
    exp_name = exp[0:-2]
    try:
      final_epoch = sorted(sorted(epochs), key=len)[-2]
      # pdb.set_trace()
    except:
      print("Not enough output {}".format(exp))
      continue
    filename = "/".join([prefix, exp, final_epoch])
    # filename = prefix + exp 
    try:
      scores = knnhomog.get_score(filename, manually_set_exp="aab")
    except:
      print("Skipping", filename)
      continue
    # pdb.set_trace()
    for label in scores:
      # pdb.set_trace()
      datas['exp'].append(exp_name)
      datas['score'].append(scores[label])
      datas['label'].append(label)
  else:
	  print("Skipping {}".format(exp))

# pdb.set_trace()

# fig = px.bar(
fig = px.box(
  data_frame=pd.DataFrame(datas),
  x='exp',
  y='score',
  color='label',
  color_discrete_sequence=px.colors.qualitative.Dark24,
  # barmode='group'
)

fig.update_layout(title="KNN Homogeneity Scores")
# fig.update_layout(xaxis={'categoryorder':'total descending'})
# fig.update_layout(xaxis={'categoryorder':'category ascending'})
fig.show()
fig.write_html('{}.html'.format(exp_filename), auto_play=False)
print("Finished writer figure to {}".format(exp_filename))
