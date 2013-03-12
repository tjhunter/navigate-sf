'''
Created on Jan 23, 2013

@author: audehofleitner
'''

import mm.arterial_hkt.pipeline_functions as pip_fun
from sklearn.linear_model import LassoLarsIC
import numpy as np
import mm.data.structures as struct
import mm.arterial_hkt.mixture as mixt
import matplotlib.pyplot as plt
import math


def detect_stops(dates, network, **param):
  """
  Main function to be run to learn the mixture distribution from the TSpot
  Input:
  tspots_data: list of list of TSpot. 
    Each list of TSpot represents a trajectory provided by the PIF
  network: network object
  """
  
  learned_mixtures = {}
  
  tspots_data = [ttob_seq for date in dates 
               for ttob_seq in pip_fun.getDayTSpotsInterpolated(date,
                                                    network)]
  tspots_cut_link = [pip_fun.seqGroupBy(tspots, keyf=lambda tsp:tsp.spot.linkId) for tspots in tspots_data]
  tspots_groups_per_link = pip_fun.groupby([link_tspots for traj_tspots in tspots_cut_link
                                    for link_tspots in traj_tspots], lambda tspots:tspots[0].spot.linkId)
  
  for lid, link_obs in tspots_groups_per_link:
    stopping_obs = ([detect_stops_on_link_traj(traj, **param)
                     if len(traj) >= 2
                     else None 
                     for traj in link_obs])
    non_stop_mix = compute_tt_from_tspots(
      [traj for i, traj in enumerate(link_obs) if stopping_obs[i] is False],
      lid,
      network,
      avg_ff_tt=None,
      **param)
    stop_mix = compute_tt_from_tspots(
      [traj for i, traj in enumerate(link_obs) if stopping_obs[i] is True],
      lid,
      network,
      avg_ff_tt=non_stop_mix[0],
      **param)
    mixt_dist = compute_mixture(non_stop_mix, stop_mix, param['non_stopping_default'])
    learned_mixtures[lid] = mixt_dist
  return learned_mixtures


def detect_mode(tspots_data, network, **param):
  """
  Main function to be run to learn the mixture distribution from the TSpot
  Input:
  tspots_data: list of list of TSpot (completed).
    Each list of TSpot represents a trajectory provided by the PIF
  network: network object
  """
  tspots_cut_link = pip_fun.seqGroupBy(tspots_data, keyf=lambda tsp:tsp.spot.linkId)
  # If there are 2 links or less traveled, then there is no full link.
  # We only compute stop go for fully traversed links 
  if len(tspots_cut_link) <= 2:
    return None
  tspots_cut_link = tspots_cut_link[1: -1]
  seqs = ([(link_tspots[0].spot.linkId,
            1 if detect_stops_on_link_traj(link_tspots, **param)
            else 0,
            (link_tspots[-1].time - link_tspots[0].time).total_seconds()) 
            for link_tspots in tspots_cut_link[1: -1]])
  return seqs if len(seqs) >= 1 else None
  

def detect_stops_on_link_traj(traj, debug=False, **param):
  """
  Detects if a trajectory (characterized by a list of tspots on a link) corresponds to a stopping vehicle
  
  Input:
  traj: list of tspots on a link
  debug (optional): if True, returns extra variables to help with debugging. Default=False
  
  Output:
  stopping (bool): if True, indicates that there is a stop on the trajectory
  If debug is set to True (default=False)
  obs: list of Point_pts representing the 
    offset and time (in seconds after beginning of traj) 
    of the measurements
  est: list of Point_pts representing the 
    offset and time (in seconds after beginning of traj) 
    of the estimated location of the measurements
  """
  if len(traj) <= 3:
    return False
  loc = np.array([x.spot.offset - traj[0].spot.offset for x in traj[1 :]])
  time = np.array([(x.time - traj[0].time).total_seconds() for x in traj])
    
  n = len(time) - 1
  A = np.zeros((n, n), dtype=np.float64)
  for i in range(n):
    A[i :, i] = time[i + 1] - time[i]
    
  model_bic = LassoLarsIC(criterion='bic', fit_intercept=False)
  # There is a bug here for the following data:
#[[  0.669923   0.         0.         0.         0.         0.         0.
#    0.      ]
# [  0.669923   2.         0.         0.         0.         0.         0.
#    0.      ]
# [  0.669923   2.        34.         0.         0.         0.         0.
#    0.      ]
# [  0.669923   2.        34.         2.         0.         0.         0.
#    0.      ]
# [  0.669923   2.        34.         2.         2.         0.         0.
#    0.      ]
# [  0.669923   2.        34.         2.         2.         4.         0.
#    0.      ]
# [  0.669923   2.        34.         2.         2.         4.         2.
#    0.      ]
# [  0.669923   2.        34.         2.         2.         4.         2.
#    0.94968 ]]
#[  6.24743444   6.24743444   6.24743444  10.41858373  14.46159935
#  26.17648665  39.90241795  52.77      ]
#
#  if A.shape == (8,8):
#    print A
#    print loc
  try:
    model_bic.fit(A, loc)
  except TypeError:
    print "Failure in detect_stops_on_link_traj"
    return False
  
  if debug:
    print 'Lasso BIC'
    print np.dot(A, model_bic.coef_)
    print model_bic.coef_
    
  stop_time = 0
  stopping = False
  for i, speed in enumerate(model_bic.coef_):
    if speed < param['speed_threshold']:
      stop_time += time[i + 1] - time[i]
      if stop_time >= param['min_stop_duration']:
        stopping = True
        break
  if not debug:
    return stopping
  est_loc = [0.0] + list(np.dot(A, model_bic.coef_))
  obs = [struct.Point_pts(s.spot.offset, t, 0) for (s, t) in zip(traj, time)]
  est = [struct.Point_pts(e + traj[0].spot.offset, t, 0) for (e, t) in zip(est_loc, time)]
  plt.plot(time[1 :], loc, '-+b', linewidth=5, label='Observations')
  plt.plot(time, est_loc, '-or', label='Estimate')
  print stopping
  plt.legend()
  plt.show()
  return (stopping, obs, est)


def compute_tt_from_tspots(link_obs, linkId, network, avg_ff_tt=None, **param):
  """
  Compute the distribution based on the observations
  
  Input:
  link_obs: array of tspots
  linkId: id of the link
  network: network object
  avg_ff_tt (optional): free flow travel time on the segment. Should be used to "complete"
    trajectories that do cover the entire link for the "slow" distribution
  
  Output:
  tuple (mean, var, tt)
  mean: estimated mean travel time
  var: estimated variance
  tt: travel times on the link, used to compute the distribution
  """
  link_length = network[linkId].length
  
  obs_tt = [(traj[-1].time - traj[0].time).total_seconds() for traj in link_obs]
  obs_dist = [traj[-1].spot.offset - traj[0].spot.offset for traj in link_obs]
  if avg_ff_tt is None:
    tt = [link_length * o_tt / o_dist for (o_tt, o_dist) in zip(obs_tt, obs_dist) if o_dist > 0]
  else:
    tt = [o_tt + avg_ff_tt * (link_length - o_dist) / link_length for (o_tt, o_dist) in zip(obs_tt, obs_dist)]
  # tt is link_length / speed on the trajectory
  if len(tt) == 0:
    return ((link_length / param['default_speed'], param['default_variance'], []) 
            if avg_ff_tt is None 
            else (avg_ff_tt + param['avg_delay'], math.pow(param['avg_delay'] / 4.0, 2), []))
  m = np.mean(tt)
  v = max(min(np.var(tt),  math.pow(m / 2.0, 2)), param['min_variance'])
  return (m, v, tt)


def compute_mixture(mix1, mix2, perc_non_stopping):
  w = np.array([len(mix1[2]), len(mix2[2])], dtype=float)
  if w.sum() <= 1e-2:
    w = np.array([perc_non_stopping, 1 - perc_non_stopping])
  w[w == 0] = 0.01
  w = w / w.sum()
  return mixt.GMixture(w, np.array([mix1[0], mix2[0]]), np.array([mix1[1], mix2[1]]))