'''
Created on Feb 1, 2013

@author: audehofleitner
'''

import numpy as np
import matplotlib.pylab as plt
import validation_plotting_functions as vpf
from mm.arterial_hkt.mixture import GMixture

np.random.seed(123456789)
K = 5
NB_SAMPLES = 100
WIDTH = .8
means = np.random.randint(-2, 5, K)
stds = np.random.rand(5)

data = np.empty((K, NB_SAMPLES))
learned_dist = np.empty((K, 10000))
mixt = [[]] * K
for i in range(K):
  data[i, :] = stds[i] * np.random.randn(NB_SAMPLES) + means[i]
  learned_dist[i, :] = stds[i] * np.random.randn(10000) + means[i]
  mixt[i] = GMixture([1.0], [np.mean(data[i, :])], [np.var(data[i, :])])

def test_scatter_box():
  plt.figure()
  bounds = {}
  for i in range(K):
    bounds[i] = np.cumsum(np.random.randint(1, 4, size=5))
  data = 15 * np.random.rand(25)
  vpf.scatter_box(bounds, data)
  plt.show()
  

def test_ksdensity_box(confidence_level):
  plt.figure()
  for i in range(K):
    plt.subplot(K, 1, i + 1)
    learn_dist = learned_dist[i, :] 
    learn_dist.sort()
    ndx = int(len(learn_dist) * (1 - confidence_level) / 2.0)
    lmr = (learn_dist[ndx], learn_dist[len(learn_dist) / 2.0], learn_dist[- ndx])
    vpf.ksdensity_box(lmr, data[i, :], mixt[i])
  plt.show()


def test_validate_dist_conf_intervals():
  for i in range(K):
    plt.figure(i)
    vpf.validate_dist_conf_intervals(data[i, :], learned_dist[i, :])
  plt.show()
  
  
def test_cumulative_scatter_box(confidence_level):
  for i in range(K):
    print 'plot', i
    plt.subplot(5,1,i+1)
    learn_dist = learned_dist[i, :] 
    learn_dist.sort()
    ndx = int(len(learn_dist) * (1 - confidence_level) / 2.0)
    lmr = (learn_dist[ndx], learn_dist[len(learn_dist) / 2.0], learn_dist[- ndx])
    points = data[i, :]
    vpf.cumulative_scatter_box(lmr, .9, points, mixt[i])
  plt.legend()
  plt.show()
  
  
test_scatter_box()
