'''
Created on Aug 27, 2012

@author: tjhunter
'''
import numpy as np
from mm.arterial_hkt.mixture import GMixture
from collections import defaultdict
from sklearn import mixture as sk_mixture
from mm.arterial_hkt.variable import VariableId
from mm.arterial_hkt.utils import tic

def getTTDistributionExact(gmrf_estimator, hmm, node_ids):
  """ Returns the travel time distribution over the link_ids, when outgoing
  with the given link.
  
  This function is the most correct and the least efficient one could do.
  
  Returns a tuple of (weighted paths, GMixture object).
  """
  # First get all the weighted paths from the HMM
  r_dis = hmm.variableDistributionFromRoute(node_ids)
  ws = np.array([w for (_, w) in r_dis])
  all_var_ids = [var_ids for (var_ids, w) in r_dis]
  # Get the mean and variance from the GMRF for each of them
  means = np.array([gmrf_estimator.pathMean(var_ids) for var_ids in all_var_ids])
  variances = np.array([gmrf_estimator.pathCovariance(var_ids) for var_ids in all_var_ids])
  return GMixture(ws, means, variances)


def getTTDistributionGivenStop(gmrf_estimator, node_ids, modes):
  """ Returns the travel time distribution over the link_ids, when outgoing
  with the given link.
  
  This function is the most correct and the least efficient one could do.
  
  Returns a tuple of (weighted paths, GMixture object).
  """
  # First get all the weighted paths from the HMM
  var_ids = [VariableId(node, mode) for (node, mode) in zip(node_ids, modes)]
  mean = gmrf_estimator.pathMean(var_ids)
  variance = gmrf_estimator.pathCovariance(var_ids)
  return GMixture([1.0], [mean], [variance])


def sequence_sampled(hmm,node_ids,trials=100,max_modes=1000000):
  res = defaultdict(int)
  for i in range(trials):
    seq = hmm.sampleVariablesFromNodes(node_ids)
    res[tuple(seq)] += 1.0/trials
  return sorted(res.items(),key=lambda x:-x[1])[:max_modes]


def getTTDistributionSampling(gmrf_estimator, hmm, node_ids, num_samples=1000, max_modes=1000000):
  """ Returns the travel time distribution over the link_ids, when outgoing
  with the given link.
  
  This function uses sampling on the modes of the HMM 
  as an approximation. The 'exact' method is getTTDistribution
  
  Returns a GMixture.
  """
  # First get all the weighted paths from the HMM
  r_dis = sequence_sampled(hmm, node_ids, num_samples, max_modes)
  ws = np.array([w for (_, w) in r_dis])
  all_var_ids = [var_ids for (var_ids, w) in r_dis]
  # Get the mean and variance from the GMRF for each of them
  means = np.array([gmrf_estimator.pathMean(var_ids) for var_ids in all_var_ids])
  variances = np.array([gmrf_estimator.pathCovariance(var_ids) for var_ids in all_var_ids])
  return GMixture(ws, means, variances)

def getTTDistributionSamplingFast(gmrf_estimator, hmm, node_ids, num_samples=1000, max_modes=1000000, precision_target=1e-2):
  """ Returns the travel time distribution over the link_ids, when outgoing
  with the given link.
  
  This function uses sampling on the modes of the HMM 
  as an approximation. The 'exact' method is getTTDistribution.
  
  This function is faster than getTTDistributionSampling: it detects when it has
  accumulated enough samples to reach some specified level of accuracy and
  stops early.
  
  Returns a GMixture.
  """
  # First get all the weighted paths from the HMM
  # Using a generator sequence
  
  # All the probability weight seen so far
  all_prob_unseen = 1.0
  all_sequences_seen = set()
  for i in range(num_samples):
    seq = tuple(hmm.sampleVariablesFromNodes(node_ids))
    # If we have not seen this sequence before, we keep it
    if seq not in all_sequences_seen:
      all_sequences_seen.add(seq)
      seq_prob = hmm.probability(seq)
      all_prob_unseen -= seq_prob
#      print "prob_unseen:",all_prob_unseen
    if len(all_sequences_seen) >= max_modes:
      break
    if all_prob_unseen <= precision_target:
      break
    if i == num_samples - 5:
      #print "getTTDistributionSamplingFast: exhausting number of samples before reaching precision"
      print "getTTDistributionSamplingFast: mixture size=% 5d, samples=% 5d, missing mass=%f"%(len(all_sequences_seen), i ,all_prob_unseen)
  all_var_ids = list(all_sequences_seen)
  ws = np.array([hmm.probability(seq) for seq in all_var_ids])
  # Renormalize the weights so that they sum to 1 exactly
  ws /= np.sum(ws)
  # Get the mean and variance from the GMRF for each of them
  means = np.array([gmrf_estimator.pathMean(var_ids) for var_ids in all_var_ids])
  variances = np.array([gmrf_estimator.pathCovariance(var_ids) for var_ids in all_var_ids])
  return GMixture(ws, means, variances)


def getTTDistribution(node_ids, gmrf_estimator, hmm, sampling_procedure, sampling_parameters=None):
  """ Returns a travel time along the sequence of node ids.
  
  Parameters:
  node_ids -- a sequence of nodeId
  gmrf_estimator -- a GMRFEstimator object
  hmm -- hmm graph object
  sampling_procedure (string) -- the name of the procedure:
      exact
      sampling
      sampling_fast
  sampling_parameters --  a dictionary
  
  Returns:
  A GMixture object
  """
  if sampling_parameters is None:
    sampling_parameters = {}
  if sampling_procedure=='exact':
    return getTTDistributionExact(gmrf_estimator, hmm, node_ids)
  if sampling_procedure=='sampling':
    return getTTDistributionSampling(gmrf_estimator, hmm, node_ids, **sampling_parameters)
  if sampling_procedure=='sampling_fast':
    return getTTDistributionSamplingFast(gmrf_estimator, hmm, node_ids, **sampling_parameters)

def learnMixture(tts, n_components, min_covar=.2):
  assert len(tts) >= n_components
  if len(tts) == 1:
    tt = tts[0]
    return GMixture(np.array([1.0]), np.array([tt]), np.array([min_covar]))
  X = np.array([tts]).T
  clf = sk_mixture.GMM(n_components=n_components, covariance_type='full', n_init=10, min_covar=min_covar)
  clf.fit(X)
  means = clf.means_.flatten()
  variances = clf.covars_.flatten()
  weights = clf.weights_.flatten()
  order = np.argsort(means)
  mix = GMixture(weights[order], means[order], variances[order])
  return mix

def learnMixtureAuto(tts, max_nb_mixtures):
  mixs = [learnMixture(tts, n) for n in range(1, min(len(tts) + 1, max_nb_mixtures + 1))]
  bics = [mix.bic(tts) for mix in mixs]
  best_idx = np.argmin(bics)
  tic("{0}:{1}".format(len(tts),best_idx+1),"learnMixtureAuto")
  return mixs[best_idx]

