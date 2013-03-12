'''
Created on Feb 4, 2013

@author: tjhunter

Input output functions to write the different part of the pipeline to some files
'''
import numpy as np
from mm.data import data_dir
from mm.data.codec_json import encode_link_id, decode_link_id
from mm.arterial_hkt.variable import VariableId
from mm.arterial_hkt.hmm import HMMNode, HMMTransition, HMMGraph
import json
from mm.arterial_hkt.gmrf import GMRF
from mm.arterial_hkt.gmrf_estimator import ExactGMRFEstimator, \
  DiagonalGMRFEstimator, JLGMRFEstimator
import time
from mm.arterial_hkt.utils import tic, s_load
try:
  import cPickle as pickle
  print "Using fast version of pickle"
except ImportError:
  import pickle
from itertools import tee


def experiment_directory(experiment_name):
  import os
  path = '%s/experiments/%s/' % (data_dir(), experiment_name)
  if not os.path.exists(path):
    print "Creating dir %s" % path
    os.makedirs(path)
  return path

def encode_NodeId(node_id):
  if type(node_id) == tuple and type(node_id[0]) == tuple:
    return {'linkId':encode_link_id(node_id[0]),
            'outgoingLinkId':encode_link_id(node_id[1])}
  else:
    return {'linkId':encode_link_id(node_id)}

def decode_NodeId(dct):
  assert len(dct) == 1
#  print "nid:",dct
  return decode_link_id(dct['linkId'])

def encode_VariableId(var_id):
  return {'nodeId':encode_NodeId(var_id.nodeId),
          'mode':var_id.mode}

def decode_VariableId(dct):
  node_id = decode_NodeId(dct['nodeId'])
  return VariableId(node_id, mode=dct['mode'])

def decode_HMMNode(dct):
  return HMMNode(decode_NodeId(dct['nodeId']), np.array(dct['modeProbs']), dct['modes'])

def encode_HMMNode(node):
  return {'nodeId':encode_NodeId(node.nodeID),
            'modeProbs':list(node.modeProbs),
            'modes':node.modes
            }

def encode_HMMTransition(trans):
  return {'fromNodeId':encode_NodeId(trans.fromNodeID),
            'toNodeId':encode_NodeId(trans.toNodeID),
            'transitions': list([list(x) for x in trans.transitions])
            }

def decode_HMMTransition(dct):
  return HMMTransition(from_node_id=decode_NodeId(dct['fromNodeId']),
                       to_node_id=decode_NodeId(dct['toNodeId']),
                       transitions=np.array(dct['transitions']))

def read_hmm(experiment_name):
  """ Reads the HMM structure
  
  Returns a HMM graph with loaded values.
  """
  f = open('%s/hmm_nodes.txt' % experiment_directory(experiment_name), 'r')
  nodes = [decode_HMMNode(dct) 
         for line in f
         for dct in [json.loads(line)]]
  f.close()
  f = open('%s/hmm_transitions.txt' % experiment_directory(experiment_name), 'r')
  transitions = [decode_HMMTransition(dct) 
         for line in f
         for dct in [json.loads(line)]]
  f.close()
  return HMMGraph(nodes, transitions) 

def save_hmm(hmm, experiment_name):
  ''' Saves the hmm data in 2 files
  
  TODO: more doc
  '''
  f = open('%s/hmm_nodes.txt' % experiment_directory(experiment_name), 'w')
  for node in hmm.allNodes():
    repr_ = {'nodeId':{
                      'linkId':{
                                'primary':node.nodeID[0],
                                'secondary':node.nodeID[1]}
                      },
            'modeProbs':list(node.modeProbs),
            'modes':node.modes
            }
    f.write(json.dumps(repr_))
    f.write('\n')
  f.close()
  del node
  f = open('%s/hmm_transitions.txt' % experiment_directory(experiment_name), 'w')
  for trans in hmm.allTransitions():
    repr_ = {'fromNodeId':{
                      'linkId':{
                                'primary':trans.fromNodeID[0],
                                'secondary':trans.fromNodeID[1]}
                      },
            'toNodeId':{
                      'linkId':{
                                'primary':trans.toNodeID[0],
                                'secondary':trans.toNodeID[1]}
                      },
            'transitions': list([list(x) for x in trans.transitions])
            }
    f.write(json.dumps(repr_))
    f.write('\n')
  f.close()

def save_gmrf_values(gmrf, experiment_name):
  f = open('%s/gmrf_means.txt' % experiment_directory(experiment_name), 'w')
  for x in gmrf.means:
    f.write(str(x))
    f.write('\n')
  f.close()
  
  f = open('%s/gmrf_upper_precision.txt' % experiment_directory(experiment_name), 'w')
  for x in gmrf.upper_precision:
    f.write(str(x))
    f.write('\n')
  f.close()

  f = open('%s/gmrf_diag_precision.txt' % experiment_directory(experiment_name), 'w')
  for x in gmrf.diag_precision:
    f.write(str(x))
    f.write('\n')
  f.close()

def save_gmrf_structure(gmrf, experiment_name):
  """ Saves the structural elements of the GMRF (can be done once with an empty GMRF)
  """
  f = open('%s/gmrf_translations.txt' % experiment_directory(experiment_name), 'w')
  sorted_var_ids = sorted(gmrf.translations.keys(), key=lambda x:gmrf.translations[x])
  for var_id in sorted_var_ids:
    repr_ = {'nodeId':{
                      'linkId':{
                                'primary':var_id.nodeId[0],
                                'secondary':var_id.nodeId[1]}
                      },
            'mode':var_id.mode
            }
    f.write(json.dumps(repr_))
    f.write('\n')
  f.close()
  
  f = open('%s/gmrf_offdiag.txt' % experiment_directory(experiment_name), 'w')
  m = len(gmrf.rows)
  for idx in range(m):
    r = gmrf.rows[idx]
    c = gmrf.cols[idx]
    f.write('%d %d\n' % (r, c))
  f.close()

def read_ttg(experiment_name):
  assert False, 'not implemented'

def save_ttg_values(tt_graph, experiment_name):
  """ Saves the values of the experiment name.
  """
  print "Saving travel time graph values in %s" % experiment_directory(experiment_name)
  fname = '%s/ttg_diag_variances.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, tt_graph.variances())
#  f = open('%s/ttg_diag_variances.txt'%experiment_directory(experiment_name),'w')
#  variances = tt_graph.variances()
#  for idx in range(tt_graph.n):
#    f.write('%f\n' % variances[idx])
#  f.close()
  
  fname = '%s/ttg_offdiag_covariances.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, tt_graph.covariances())
#  f = open('%s/ttg_offdiag_covariances.txt'%experiment_directory(experiment_name),'w')
#  covariances = tt_graph.covariances()
#  for idx in range(tt_graph.m):
#    f.write('%f\n' % covariances[idx])
#  f.close()
  
  fname = '%s/ttg_variable_counts.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, tt_graph.variableCounts(),fmt='%d')

  fname = '%s/ttg_edge_counts.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, tt_graph.edgeCounts(),fmt='%d')

def save_ttg_structure(tt_graph, experiment_name):
  """ Saves the structural elements of the TT Graph (can be done once with an empty GMRF)
  """
  print "Saving travel time graph structure in %s" % experiment_directory(experiment_name)
  f = open('%s/ttg_translations.txt' % experiment_directory(experiment_name), 'w')
  for var_id in tt_graph.reverse_indexes:
    repr_ = encode_VariableId(var_id)
    f.write(json.dumps(repr_))
    f.write('\n')
  f.close()
  
  f = open('%s/ttg_offdiag_translations.txt' % experiment_directory(experiment_name), 'w')
  for (from_var_id, to_var_id) in tt_graph.reverse_edge_indexes:
    repr1_ = encode_VariableId(from_var_id)
    repr2_ = encode_VariableId(to_var_id)
    repr_ = [repr1_, repr2_]
    f.write(json.dumps(repr_))
    f.write('\n')
  f.close()

  f = open('%s/ttg_offdiag.txt' % experiment_directory(experiment_name), 'w')
  for idx in range(tt_graph.m):
    r = tt_graph.rows()[idx]
    c = tt_graph.cols()[idx]
    f.write('%d %d\n' % (r, c))
  f.close()

def read_gmrf(experiment_name):
  f = open('%s/ttg_translations.txt' % experiment_directory(experiment_name), 'r')
  reverse_translations = [decode_VariableId(dct) 
                          for line in f
                          for dct in [json.loads(line)]]
  translations = dict(zip(reverse_translations, range(len(reverse_translations))))
  
  f = open('%s/ttg_offdiag.txt' % experiment_directory(experiment_name), 'r')
  rowcols = [[int(x) for x in line.split()] for line in f]
  rows = np.array([row for (row, _) in rowcols], dtype=np.int)
  cols = np.array([col for (_, col) in rowcols], dtype=np.int)
  f = open('%s/gmrf_means.txt' % experiment_directory(experiment_name), 'r')
  means = np.array([float(line) for line in f], dtype=np.double)
  f = open('%s/gmrf_upper_precision.txt' % experiment_directory(experiment_name), 'r')
  upper_precision = np.array([float(line) for line in f], dtype=np.double)
  f = open('%s/gmrf_diag_precision.txt' % experiment_directory(experiment_name), 'r')
  diag_precision = np.array([float(line) for line in f], dtype=np.double)
  return GMRF(translations, rows, cols, means, diag_precision, upper_precision)


def save_gmrf_estimator_JLGMRFEstimator(gmrf_estimator, experiment_name):
  fname = '%s/gmrf_estimator_Q.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, gmrf_estimator.Q)

def save_gmrf_estimator_ExactGMRFEstimator(gmrf_estimator, experiment_name):
  assert type(gmrf_estimator) == ExactGMRFEstimator
  fname = '%s/gmrf_estimator_covariance.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, gmrf_estimator._covariance)
  
def read_translation(experiment_name):
  f = open('%s/ttg_translations.txt' % experiment_directory(experiment_name), 'r')
  reverse_translations = [decode_VariableId(dct) 
                          for line in f
                          for dct in [json.loads(line)]]
  translation = dict(zip(reverse_translations, range(len(reverse_translations))))
  return translation

def save_gmrf_estimator_DiagonalGMRFEstimator(gmrf_estimator, experiment_name):
  assert type(gmrf_estimator) == DiagonalGMRFEstimator
  fname = '%s/gmrf_estimator_diag_variance.txt' % experiment_directory(experiment_name)
  np.savetxt(fname, gmrf_estimator.diag_variance)

def read_gmrf_estimator_JLGMRFEstimator(experiment_name):
  fname = '%s/gmrf_estimator_Q.txt' % experiment_directory(experiment_name)
  Q = np.loadtxt(fname)
  translation = read_translation(experiment_name)
  f = open('%s/gmrf_means.txt' % experiment_directory(experiment_name), 'r')
  means = np.array([float(line) for line in f], dtype=np.double)
  return JLGMRFEstimator(translation, means, Q)

def read_gmrf_estimator_DiagonalGMRFEstimatorr(experiment_name):
  fname = '%s/gmrf_estimator_diag_variance.txt' % experiment_directory(experiment_name)
  diag_variance = np.loadtxt(fname)
  translation = read_translation(experiment_name)
  f = open('%s/gmrf_means.txt' % experiment_directory(experiment_name), 'r')
  means = np.array([float(line) for line in f], dtype=np.double)
  return DiagonalGMRFEstimator(translation, means, diag_variance)

def read_gmrf_estimator_ExactGMRFEstimator(experiment_name):
  fname = '%s/gmrf_estimator_covariance.txt' % experiment_directory(experiment_name)
  cov = np.loadtxt(fname)
  translation = read_translation(experiment_name)
  f = open('%s/gmrf_means.txt' % experiment_directory(experiment_name), 'r')
  means = np.array([float(line) for line in f], dtype=np.double)
  return ExactGMRFEstimator(translation, means, cov)

def get_gmrf_estimator(experiment_name, process):
  print 'Loading gmrf estimator: experiment={0}, process={1}'.format(experiment_name, process)
  if process == 'diagonal':
    gmrf_estimator = read_gmrf_estimator_DiagonalGMRFEstimatorr(experiment_name)
  if process == 'jl':
    gmrf_estimator = read_gmrf_estimator_JLGMRFEstimator(experiment_name)
  if process == 'exact':
    gmrf_estimator = read_gmrf_estimator_ExactGMRFEstimator(experiment_name)
  return gmrf_estimator

# pylint:disable=W0142
def test_traj_obs(experiment_name,print_counter=1000):
  fname_test = "%s/traj_obs_test.pkl"%experiment_directory(experiment_name)
  tic("test_traj_obs: opening test trajectory obs in %s"%fname_test, experiment_name)
  f = open(fname_test, 'r')
  c = 0
  for traj_ob in s_load(f):
    c += 1
    if print_counter > 0 and c % print_counter == 0:
      tic("test_traj_obs: Consumed so far {0} observations".format(c), experiment_name)
    yield traj_ob

def read_hmm_pickle(experiment_name):
  hmm_graph_fname = "%s/hmm_graph.pkl"%experiment_directory(experiment_name)
  tic("Reading hmm from %s"%hmm_graph_fname, experiment_name)
  return pickle.load(open(hmm_graph_fname,'r'))
