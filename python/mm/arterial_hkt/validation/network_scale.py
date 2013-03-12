'''
Created on Feb 11, 2013

@author: audehofleitner
'''

from mm.arterial_hkt.mixture_functions import getTTDistribution, \
  getTTDistributionGivenStop
import numpy as np
from collections import defaultdict
from scipy.integrate import simps


def model_validation(data,
                     gmrf_est,
                     hmm,
                     net,
                     confidence_levels,
                     given_mode,
                     estimation_sampling_process,
                     estimation_sampling_parameters,
                     **param):
  ll = defaultdict(list)
  conf = np.zeros(len(confidence_levels), dtype=float)
  percentile = np.zeros(len(confidence_levels), dtype=float)
  length_bin_size = param['length_bin_size']
  max_nb_paths = param['max_nb_paths']
  for traj_obs in data[: max_nb_paths]:
    tt = sum([obs.value for obs in traj_obs.observations])
    length = sum([net[obs.varId.nodeId].length for obs in traj_obs.observations])
    if given_mode:
      dist = getTTDistributionGivenStop(gmrf_est,
                                        [obs.varId.nodeId for obs in traj_obs.observations],
                                        [obs.varId.mode for obs in traj_obs.observations])
    else:
      dist = getTTDistribution([obs.varId.nodeId for obs in traj_obs.observations],
                               gmrf_est,
                               hmm,
                               sampling_procedure=estimation_sampling_process,
                               sampling_parameters=estimation_sampling_parameters)
    ll[int(length / length_bin_size)] += [dist.logProbability(tt)]
    (c, q) = tt_bound_quantiles(tt, dist, confidence_levels)
    conf += c
    percentile += q 
  conf = conf / float(len(data[: max_nb_paths]))
  conf = np.hstack(([0], conf, [1]))
  percentile = percentile / float(len(data[: max_nb_paths]))
  percentile = np.hstack(([0], percentile, [1]))
  confidence_levels = np.hstack(([0], confidence_levels, [1]))
  
  conf_up_area = np.max(np.vstack((conf - confidence_levels, np.zeros_like(conf))), axis=0)
  conf_down_area = np.max(np.vstack((confidence_levels - conf, np.zeros_like(conf))), axis=0)
  conf_up_area = simps(conf_up_area, confidence_levels)
  conf_down_area = simps(conf_down_area, confidence_levels)
  
  percentile_up_area = np.max(np.vstack((percentile - confidence_levels, np.zeros_like(percentile))), axis=0)
  percentile_down_area = np.max(np.vstack((confidence_levels - percentile, np.zeros_like(percentile))), axis=0)
  percentile_up_area = simps(percentile_up_area, confidence_levels)
  percentile_down_area = simps(percentile_down_area, confidence_levels)
  
  ll_res = {}
  for b, ll_val in ll.items():
    if len(ll_val) > param['min_nb_validation_points']:
      ll_res[b] = (np.median(ll_val), np.std(ll_val))
  return ll_res, [conf, conf_up_area, conf_down_area], [percentile, percentile_up_area, percentile_down_area] 

  
def tt_bound_quantiles(tt,
             dist,
             confidence_levels):
  val_low = ((1 - confidence_levels) / 2.0)
  val_up = (1 - ((1 - confidence_levels) / 2.0))
  ival = dist.cumulative(tt)
  b = [(u >= ival and l <= ival) for (l,u) in zip(val_low, val_up)]
  q = [c >= ival for c in confidence_levels]
  return b, q

def tt_bound(tt,
             dist,
             confidence_levels):
  val_low = ((1 - confidence_levels) / 2.0)[::-1]
  val_up = (1 - ((1 - confidence_levels) / 2.0))[::-1]
  bound = [None, None]
  def is_within_bound(l, u, tt):
    bound[0] = dist.inverseCumulative(l, low_bound=bound[0], up_bound=bound[1])
    bound[1] = dist.inverseCumulative(u, low_bound=bound[0], up_bound=bound[1])
    return tt >= bound[0] and tt <= bound[1]
  r = map(lambda (l, u): is_within_bound(l, u, tt),
             zip(val_low, val_up))
  list.reverse(r)
  return r
