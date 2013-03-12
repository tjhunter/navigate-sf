'''
Created on Jan 29, 2013

@author: audehofleitner
'''

import numpy as np
from collections import defaultdict
import operator


def number_measurements_per_path(traj_obs, net, debug=False, **param):
  """ Counts the number of measurements per subpath in the dataset.
  Input:
  traj_obs: list of TrajectoryObservation
  net: network object
  debug: parameter to print more info to screen for debugging/testing
  param: parameters set in pipeline_script_#.py
    min_validation_path_length: minimum length of subpaths to be
      considered. If the param is not set, all paths are considered
  Output:
  dictionary of subpaths and corresponding counts
  """
  
  path_length = (param['min_validation_path_length'] 
                 if 'min_validation_path_length' in param 
                 else 0)
  
  nb_obs_per_path = defaultdict(int)
  for traj_ob in traj_obs:
    if len(traj_ob.observations) == 0:
      continue
    link_lengths = np.array([net[o.varId.nodeId].length for o in traj_ob.observations])
    node_ids = [o.varId.nodeId for o in traj_ob.observations]
    cum_length = np.cumsum(link_lengths)
    start_ndx = 0
    while cum_length[-1] > path_length:
      for ndx in [i for i, c in enumerate(cum_length) if c > path_length]:
        nb_obs_per_path[tuple(node_ids[start_ndx : ndx + 1])] += 1
      cum_length -= link_lengths[start_ndx]
      start_ndx += 1
  
  if debug:
    validation_paths = nb_obs_per_path.items()
    validation_paths.sort(key=operator.itemgetter(1), reverse=True)
    for route, nb in validation_paths:
      s = str([r for r in route]) + str(nb)
      print s + '\n'
  return nb_obs_per_path.items()


def select_validation_paths(data,
                           net,
                           debug=False,
                           **param):
  """ Selects subpaths based on the nb of measurements on each subpath
  Input: 
  data: list of TrajectoryObservation
  net: network object
  param: script parameters
  Output:
  list containing lists of nodeId which define the validation paths
  """
  
  nb_obs_per_path = number_measurements_per_path(data, net, debug, **param)
  
  min_nb_path = param['min_nb_validation_points']
  if debug:
    obs_per_path = nb_obs_per_path[:]
    obs_per_path.sort(key=operator.itemgetter(1), reverse=True)
    print obs_per_path
  validation_paths = ([obs_per_path 
                       for obs_per_path in nb_obs_per_path 
                       if obs_per_path[1] >= min_nb_path])
  if debug:
    obs_per_path = validation_paths[:]
    obs_per_path.sort(key=operator.itemgetter(1), reverse=True)
    print obs_per_path
  max_nb_paths = param['max_nb_validation_paths']
  if len(validation_paths) > max_nb_paths:
    validation_paths.sort(key=operator.itemgetter(1), reverse=True)
  res = []
  for v in validation_paths:
    if len(res) >= max_nb_paths:
      break
    add_path = True
    for r in res:
      if (len(set(v[0]) - set(r[0])) <= 0.2 * len(v[0]) 
          or len(set(r[0]) - set(v[0])) <= 0.2 * len(r[0])):
        add_path = False
        break
    if add_path:
      res.append(v[0])
  return res


def select_validation_data_given_paths(data, val_paths, debug=False):
  validation_data = defaultdict(list)
  def needle_haystack(needle, haystack):
    l1, l2 = len(haystack), len(needle)
    for i in range(l1 - l2 + 1):
      if haystack[i : i + l2] == needle:
        return i
    return None
  for traj_ob in data:
    path_ob = tuple([o.varId.nodeId for o in traj_ob.observations])
    for val_path in val_paths:
      ndx = needle_haystack(val_path, path_ob)
      if ndx is not None:
        validation_data[val_path] += [traj_ob.observations[ndx: ndx + len(val_path)]]
  return validation_data
