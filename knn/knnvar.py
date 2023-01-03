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

track_classes = True

prefix = sys.argv[1]

datas = {
  'size': [],
  'score': [],
  'label': []
}

for size in tqdm(os.listdir(prefix)):
  # pdb.set_trace()
  # dumb mac .DS_Store thing
  if "/." not in prefix + "/" + size:
    runs = os.listdir(prefix + "/" + size)
    for run in runs:
      if "/." not in prefix + "/" + size + '/' + run:
        epochs = os.listdir(prefix + "/" + size + '/' + run)
        try:
          final_epoch = sorted(sorted(epochs), key=len)[-2]
        except:
          print("Not enough output {}".format(size + ' ' + run))
          continue
        # pdb.set_trace()
        scores = knnhomog.get_score("/".join([
          prefix,
          size,
          run,
          final_epoch
        ]), manually_set_exp='nometa_nested')
        for label in scores:
          datas['size'].append(size)
          datas['score'].append(scores[label])
          datas['label'].append(label)
      else:
          print("Skipping {}".format(size))

fig = px.box(
  data_frame=pd.DataFrame(datas),
  x='size',
  y='score',
  color='label',
  color_discrete_sequence=px.colors.qualitative.Dark24,
  # barmode='group'
)

fig.update_layout(title="KNN Homogeneity Scores by Network Size")
# fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_layout(xaxis={'categoryorder':'category ascending'})
fig.show()
fig.write_html('temp.html', auto_play=False)
print("Finished writer figure")