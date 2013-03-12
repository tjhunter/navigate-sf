'''
Created on Jan 24, 2013

@author: audehofleitner
'''

import detect_stops as ds
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
from mm.data import get_network
import mm.arterial_hkt.pipeline_functions as pip_fun
from mm.arterial_hkt.pipeline_script_2 import (trajectory_conversion, data_source, graph_type, experiment_design)

def test_detect_stops_on_link_traj(traj, **param):
  (stopping, obs, est) = ds.detect_stops_on_link_traj(traj, debug=True, **param)
  print stopping
  
  if param['display']:
    plt.figure(1, figsize=(7, 5))
    plt.plot([o.time for o in obs], [o.space for o in obs], 'r-', lw=2)
    plt.plot([o.time for o in est], [o.space for o in est], 'b-', lw=2)
    plt.show()
    

def test_compute_tt_from_tspots(list_traj, lid, network, avg_ff_tt=None, **param):
  mix = ds.compute_tt_from_tspots(list_traj, lid, network, avg_ff_tt, **param)
  if param['display']:
    plt.figure(1, figsize=(7, 5))
    mu = mix[0]
    sigma = np.sqrt(mix[1])
    tt = mix[2]
    if len(tt) < 2:
      print 'not enough measurements'
      print mu
      print sigma
      return mix
    print mu
    print sigma
    print tt
    plt.hist(tt, normed=True)
    min_val = min(min(tt) - 10, mu - 2 * sigma)
    max_val = max(max(tt) + 10, mu + 2 * sigma)
    x = np.linspace(min_val, max_val, 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    plt.show()
  return mix
  
  

if __name__ == '__main__':
  basic_geometry = experiment_design['basic_geometry']
  net = get_network(**basic_geometry)
  graph_type=experiment_design['graph_type']
  data_source = experiment_design['data_source']
  dates = experiment_design['data_source']['dates']
  basic_geometry = experiment_design['basic_geometry']

  tspots_seqs = [ttob_seq for date in dates 
               for ttob_seq in pip_fun.getDayTSpots(data_source['feed'], 
                                                    basic_geometry['nid'],
                                                    date, 
                                                    basic_geometry['net_type'],
                                                    basic_geometry['box'], 
                                                    net)]
  
  tspots_cut_link = [pip_fun.seqGroupBy(tspots, keyf=lambda tsp:tsp.spot.linkId) for tspots in tspots_seqs]
  tspots_groups_per_link = pip_fun.groupby([link_tspots for traj_tspots in tspots_cut_link
                                    for link_tspots in traj_tspots], lambda tspots:tspots[0].spot.linkId)
  param = trajectory_conversion['params']
  param['display'] = False
  for lid, link_obs in tspots_groups_per_link:
    stopping_obs = ([ds.detect_stops_on_link_traj(traj, **param) 
                     if len(traj) >= 2 
                     else None 
                     for traj in link_obs])
    print 'Non stopping'
    non_stop_mix = test_compute_tt_from_tspots(
      [traj for i, traj in enumerate(link_obs) if stopping_obs[i] is False],
      lid,
      net,
      avg_ff_tt=None,
      **param)
    print 'Stopping'
    stop_mix = test_compute_tt_from_tspots(
      [traj for i, traj in enumerate(link_obs) if stopping_obs[i] is True],
      lid,
      net,
      avg_ff_tt=non_stop_mix[0],
      **param)
    # mixt_dist = compute_mixture(non_stop_mix, stop_mix)

                 
  