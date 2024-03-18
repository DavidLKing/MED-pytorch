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

datas = {
  'exp': [],
  'score': [],
  'label': []
}

for exp in tqdm(os.listdir(prefix)):
  # dumb mac .DS_Store thing
  if "/." not in prefix + "/" + exp and 'nested' not in exp:
    # pdb.set_trace()
    epochs = os.listdir(prefix + "/" + exp)
    # epochs = os.listdir(prefix)
    try:
      final_epoch = sorted(sorted(epochs), key=len)[-2]
    except:
      print("Not enough output {}".format(exp))
      continue
    filename = "/".join([prefix, exp, final_epoch])
    # filename = prefix + exp 
    scores = knnhomog.get_score(filename, manually_set_exp="aab")
    for label in scores:
      datas['exp'].append(exp)
      datas['score'].append(scores[label])
      datas['label'].append(label)
  else:
	  print("Skipping {}".format(exp))

fig = px.bar(
  data_frame=pd.DataFrame(datas),
  x='exp',
  y='score',
  color='label',
  color_discrete_sequence=px.colors.qualitative.Dark24,
  barmode='group'
)

fig.update_layout(title="KNN Homogeneity Scores")
# fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_layout(xaxis={'categoryorder':'category ascending'})
# fig.show()
fig.write_html('temp.html', auto_play=False)
print("Finished writer figure")
