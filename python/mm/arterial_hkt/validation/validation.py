'''
Created on Feb 4, 2013

@author: audehofleitner
'''
# Make sure all imports are absolute
import mm.arterial_hkt.validation.validation_paths as vp
from mm.arterial_hkt.mixture_functions import getTTDistribution
import matplotlib.pyplot as plt
import mm.arterial_hkt.validation.validation_plotting_functions as vpf
import mm.arterial_hkt.validation.network_scale as net_val
import math
import numpy as np
from collections import defaultdict
import os

def get_project_root():
  """ Some users (Tim) run scripts using ipython from the python/ 
  directory, while others use pydev, which sets the cwd to the local script.
  This script extracts the absolute path of the project (assuming it got downloaded from 
  git and that the interpreter is running in a subdirectory)
  """
  p = os.getcwd()
  s = p.split("navigate-sf")[0] + "/navigate-sf/"
  print "Project root is ",s
  return s

FIGURES_DIR = '%s/reports/kdd13/figs-plots-tmp/' % get_project_root()

def validate_on_paths(models,
                      net,
                      estimation_sampling_process,
                      estimation_sampling_parameters,
                      min_validation_path_lengths=[],
                      **param):
  """ Main function which computes validate_on_paths metrics and plots
  Input:
  models: list of tuples where each tuple is (data, gmrf, hmm, name)
  net: network object
  estimation_sampling_process: string, see getTTDistribution Found at: mm.arterial_hkt.mixture_functions
  estimation_sampling_parameters: dict, see getTTDistribution Found at: mm.arterial_hkt.mixture_functions
  param: constant variables defined in pipeline_script_#.py
  """
  # Make sure there is only one path specified
  assert not min_validation_path_lengths or ("min_validation_path_length" not in param)
  if min_validation_path_lengths:
    validation_paths = []
    # Get one validation path for each of the elements
    for min_validation_path_length in min_validation_path_lengths:
      param2 = dict(param.items()+[("min_validation_path_length",min_validation_path_length)])
      validation_paths += vp.select_validation_paths(models[0][0], net, debug=False, **param2)
  else:
    validation_paths = vp.select_validation_paths(models[0][0], net, debug=False, **param)
  print "Len val paths,",len(validation_paths)
  learned_dist = defaultdict(dict)
  validation_tt = defaultdict(list)
  lmrs = defaultdict(dict)
  
  for i, (all_data, gmrf, hmm, name) in enumerate(models):
    validation_data = vp.select_validation_data_given_paths(all_data, validation_paths)
    for val_path, data in validation_data.iteritems():
      gmix = getTTDistribution(list(val_path),gmrf, hmm,
                               sampling_procedure=estimation_sampling_process,
                               sampling_parameters=estimation_sampling_parameters)
      learned_dist[val_path][name] = gmix
      
      if i == 0:
        val_tt = map(lambda d: sum([obs.value for obs in d]), data)
        validation_tt[val_path] = val_tt
      
      bounds = (gmix.inverseCumulative(0.05),
             gmix.inverseCumulative(0.25),
             gmix.inverseCumulative(0.5),
             gmix.inverseCumulative(0.75),
             gmix.inverseCumulative(0.95))
      lmrs[val_path][name] = bounds
  plot_scatter_box(lmrs, validation_tt)
  plot_pdf(validation_tt, learned_dist)
  plot_cdf(validation_tt, learned_dist, **param)
  plot_pp_plot(validation_tt, learned_dist)
  plot_qq_plot(validation_tt, learned_dist)

def validate_on_network(models,
                        net,
                        confidence_levels=np.linspace(0.05, 0.95, 15),
                        estimation_sampling_process=None,
                        estimation_sampling_parameters=None,
                        **param):
  ll = {}
  conf = {}
  percentile = {}
  for (data, gmrf, hmm, name) in models:
    print name
    # For some reason, ipython does not like breaking the line
    (ll[name], conf[name], percentile[name]) = net_val.model_validation(data,
                                                 gmrf,
                                                 hmm,
                                                 net,
                                                 confidence_levels,
                                                 given_mode=False,
                                                 estimation_sampling_process=estimation_sampling_process,
                                                 estimation_sampling_parameters=estimation_sampling_parameters,
                                                 **param)
  #    (ll['{0} mode known'.format(name)],
  #     conf['{0} mode known'.format(name)]) = net_val.model_validation(data,
  #                                                                    gmrf,
  #                                                                    hmm,
  #                                                                    confidence_levels,
  #                                                                    given_mode=True)
  # Putting some indexes on figures to enable multiple plots
  plt.figure(100, figsize=(8,4))
  vpf.validate_intervals(np.hstack(([0], confidence_levels, [1])),
                                   conf)
  plt.xlabel('Level of confidence of the interval')
  plt.ylabel('Perc. of obs. in the interval')
  save_fig('interval_val.pdf')
  
  plt.figure(101, figsize=(8,4))
  vpf.validate_intervals(np.hstack(([0], confidence_levels, [1])),
                                   percentile)
  plt.xlabel('Percentile')
  plt.ylabel('Perc. of obs. in the percentile')
  save_fig('percentile_val.pdf')
  
  plt.figure(102, figsize=(8,4))
  vpf.likelihood_dist_per_bins(ll, param['length_bin_size'])
  save_fig('ll_val.pdf')
    
    
def plot_scatter_box(bounds, val_data, **param):
  fig = plt.figure(figsize=(5,6))
  nb_fig = len(bounds)
  n_rows = math.ceil(nb_fig / 2.0)
  for i, (val_path, data) in enumerate(val_data.items()):
    ax = fig.add_subplot(n_rows, 2, i + 1)
    vpf.scatter_box(bounds[val_path], data)
    ax.text(plt.xlim()[0] + 2, 0.8, '({0})'.format(i))
    if i >= nb_fig - 2:
      plt.xlabel('Travel time (s)', fontsize=10)
#      # Making sure we do not have negative tts
#    (low,high) = plt.xlim()
#    plt.xlim((max(low,0),high))
  ax.legend(bbox_to_anchor=(1.2, -0.22), loc='lower left',
            borderaxespad=0.,
            prop={'size':8})
  save_fig('scatter_box.pdf')
  

def plot_pdf(val_data, distributions, **param):
  fig = plt.figure(figsize=(5,9))
  nb_fig = len(val_data)
  n_rows = math.ceil(nb_fig / 2.0)
  for i, (val_path, data) in enumerate(val_data.items()):
    ax = fig.add_subplot(n_rows, 2, i + 1)
    vpf.ksdensity_(data, distributions[val_path])
    # plt.title('Path {0}'.format(i))
    ymin, ymax = plt.ylim()
    ax.text(0.05, np.mean([ymin, ymax]), 'Path {0}'.format(i))
    if i % 2 == 0:
      plt.ylabel('$P_X(x)$')
    if i >= nb_fig - 2:
      plt.xlabel('Travel time (s)')
    ax.locator_params(nbins=4)
  ax.legend(bbox_to_anchor=(1.2, 0), loc='lower left', borderaxespad=0.)
  save_fig('pdf_val.pdf')
  
  
def plot_cdf(val_data, distributions, **param):
  fig = plt.figure(figsize=(5,5.5))
  nb_fig = len(val_data)
  n_rows = math.ceil(nb_fig / 2.0)
  for i, (val_path, data) in enumerate(val_data.items()):
    ax = fig.add_subplot(n_rows, 2, i + 1)
    vpf.cumulative(data, distributions[val_path])
    # plt.title('Path {0}'.format(i))
    ax.text(0.15, 0.8, '({0})'.format(i))
    if i % 2 == 0:
      plt.ylabel('$\mathbf{P}(Z^{(p)} \leq t)$')
    if i >= nb_fig - 2:
      plt.xlabel('Travel time $t$ (s)', fontsize=10)
    ax.locator_params(nbins=4)
  ax.legend(bbox_to_anchor=(1.2, -0.22),
            loc='lower left',
            borderaxespad=0.,
            prop={'size':8})
  save_fig('cdf_val.pdf')
  
  
def plot_pp_plot(val_data, distributions):
  fig = plt.figure()
  nb_fig = len(val_data)
  n_rows = math.ceil(nb_fig / 2.0)
  for i, (val_path, data) in enumerate(val_data.items()):
    ax = fig.add_subplot(n_rows, 2, i + 1)
    vpf.pp_plot(data, distributions[val_path])
    # plt.title('Path {0}'.format(i))
    ax.text(0.1, 0.8, 'Path {0}'.format(i))
    if i % 2 == 0:
      plt.ylabel('Empirical cdf')
    if i >= nb_fig - 2:
      plt.xlabel('Learned cdf')
    ax.locator_params(nbins=4)
  ax.legend(bbox_to_anchor=(1.2, -0.25), loc='lower left', borderaxespad=0.)
  save_fig('paths_pp_plot.pdf')
  
  
def plot_qq_plot(val_data, distributions):
  fig = plt.figure()
  nb_fig = len(val_data)
  n_rows = math.ceil(nb_fig / 2.0)
  for i, (val_path, data) in enumerate(val_data.items()):
    ax = fig.add_subplot(n_rows, 2, i + 1)
    vpf.qq_plot(data, distributions[val_path])
    # plt.title('Path {0}'.format(i))
    ax.text(ax.get_xlim()[0] + 5, ax.get_ylim()[1] - 15, 'Path {0}'.format(i))
    if i % 2 == 0:
      plt.ylabel('Empirical quantiles')
    if i >= nb_fig - 2:
      plt.xlabel('Learned quantiles')
    ax.locator_params(nbins=4)
  ax.legend(bbox_to_anchor=(1.2, -0.25), loc='lower left', borderaxespad=0.)
  save_fig('paths_qq_plot.pdf')
  
def save_fig(figname):
  plt.savefig('{0}/{1}'.format(FIGURES_DIR, figname), bbox_inches='tight')
  
