

import imageio
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import collections
import numpy as np
import scipy.stats as stats
import math
from slicer_func import *
import  seaborn as  sns


def labels_gif(X: np.array,labels_array: np.array, title: str, filename: str, pompom = False ):
  '''Creates a gif from all the labelings visited by the algorithms'''

  images_dir = 'images'
  img = plt.imread("pompom.jpeg")
  time = range(0,len(labels_array))

  def create_frame(t):
    fig = plt.figure(figsize=(6, 6))
    if pompom:
      plt.imshow(img, extent=[0, 8, -1, 7])
    plt.scatter(X[:, 0], X[:, 1], c=labels_array[t], s=40, cmap='gist_rainbow')
    plt.title(f'{title}, Iter: {t}',
              fontsize=14)
    plt.savefig(f'{images_dir}/frames/img_{t}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()

  for t in time:
    create_frame(t)

  frames = []
  for t in time:
      image = imageio.imread(f'{images_dir}/frames/img_{t}.png')
      frames.append(image)

  imageio.mimsave(f'{images_dir}/gifs/{filename}.gif', # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second
  

def trace_df(cluster_centers,X):
  '''Creates a trace plot of a specified parameter'''
  outlier_center = outlier_region(X)
  df = pd.DataFrame(columns=["iter","cluster","center_x1","center_x2"])
  for t in range(len(cluster_centers)):
    centers = np.stack(cluster_centers[t],axis=0)
    for i in range(len(centers)):
      df.loc[len(df)] = [t,i,centers[i][0],centers[i][1]]
  df = df[df['center_x1']<outlier_center[0]]

  return df

def plot_traceplot(cluster_centers,X, title):
  df = trace_df(cluster_centers,X)
  for i in range(1,3):
    g = sns.scatterplot(data=df,x="iter",y=f"center_x{i}",hue="cluster",palette='gist_rainbow',legend=False,).set(title=f'{title}')
    images_dir = 'images'
    plt.savefig(f'{images_dir}/trace_plots/{title}_coordinate{i}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()
